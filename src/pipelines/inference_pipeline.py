import torch
from loaders.loader import load_models
from models.msg_spade_decoder import MSGSpadeDecoder
from face_parsing.models.utils import get_mask_by_idx
from hairshifter.utils import strip_ddp_prefix


class InferencePipeline:
    """
    InferencePipeline:
    - provides inference()
    - provides load_checkpoint()
    """

    def __init__(self, device, loaded=True):
        self.device = device

        # --- Generator (G) ---
        self.E_H = load_models("E_H", pretrained=loaded,
                               freeze=True).to(self.device)
        self.E_M = load_models("E_M", pretrained=loaded,
                               freeze=True).to(self.device)
        self.E_C = load_models("E_C", pretrained=False).to(
            self.device)  # trainable
        self.W = load_models("W", pretrained=loaded,
                             freeze=True).to(self.device)
        self.M_C = load_models("M_C", pretrained=True,
                               freeze=True).to(self.device)

        # Same upscale as training: the trained output head is conv_img (64->3).
        # Upscaling to frame_size happens in inference.py
        # D_S will load and freeze all params except for GF_SPADEs
        self.D_S = load_models("D_S", pretrained=loaded,
                               strict=False, freeze=True).to(self.device)
        self.D_C = load_models("D_C", pretrained=loaded,
                               freeze=True).to(self.device)
        self.D = MSGSpadeDecoder(self.D_C, self.D_S)

        # E_C has BatchNorm and is never put in eval by load_models
        for module in (self.E_C, self.D_S, self.D_C):
            module.eval()

        self.modules_to_load = {
            "E_C": self.E_C,
            "D_S": self.D_S,
        }

        # Not frozen yet but will be used in inference mode
        self.synthesizer_decoder_trainable_dict = {
            k: p for k, p in self.D_S.named_parameters() if p.requires_grad}
        self.context_encoder_trainable_dict = {
            k: p for k, p in self.E_C.named_parameters() if p.requires_grad}
        self.generator_trainable_dict = self.synthesizer_decoder_trainable_dict | \
            self.context_encoder_trainable_dict

    def set_ema(self, ema):
        self.ema = ema

    @torch.no_grad()
    def inference(self, I_s, I_d_t):
        # TODO: should I send the image right back or should I make it into an entire video ?
        # For the pipeline let's just send the image back
        I_s = I_s.to(self.device)
        I_d_t = I_d_t.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            f_c = self.E_C(I_d_t)
            f_h = self.E_H(I_s)

            f_m = self.E_M(I_s)["kp"].view(I_s.size(0), -1, 3)
            f_m_d = self.E_M(I_d_t)["kp"].view(I_s.size(0), -1, 3)

            f_w = self.W(feature_3d=f_h, kp_source=f_m,
                         kp_driving=f_m_d)['out']

            # Same hair mask as training (class 17 on the context image)
            m_c = get_mask_by_idx(I_d_t, self.M_C,
                                  device=self.device, class_idx=17)
            I_p = self.D(f_c, f_w, m_c)

            return I_p

    def load_checkpoint(self, path: str):
        """
        Load checkpoint into respective modules. Returns checkpoint dict.
        """
        ck = torch.load(path, map_location=self.device)
        for name, module in self.modules_to_load.items():
            # strict: a silent partial load gives gibberish output
            module.load_state_dict(strip_ddp_prefix(ck[name]))

        if "ema" in ck:
            ema = strip_ddp_prefix(ck["ema"])
            for name, param in self.generator_trainable_dict.items():
                if param.requires_grad and name in ema:
                    self.ema.shadow[name] = ema[name].clone()

        return ck
