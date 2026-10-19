"""Bản vá `LLaVAMedAdapter` của AMANDA — sửa lỗi cắt mất đầu câu trả lời.

⛔ KHÔNG sửa `third_party/AMANDA/`. Vá bằng **kế thừa lớp**, ghi đè đúng MỘT phương thức.

## Bug

`LLaVAMedAdapter._decode_output` giả định `model.generate()` trả về **prompt + đáp án**, nên cắt
`output_ids[0, input_token_len:]` để bỏ prompt:

    input_ids  = 18 token · output_ids = 25 token
    giải mã đầy đủ   : "Yes, the image is a CT scan of the chest, which includes the lungs."
    AMANDA cắt [18:] : "which includes the lungs."          ← mất cả chữ "Yes"

Nhưng LLaVA-Med truyền `inputs_embeds` cho `generate()` (vì phải chèn 576 patch embedding của ảnh),
và khi dùng `inputs_embeds` thì HF **chỉ trả về token MỚI**. ⇒ Phép cắt chặt mất `input_token_len`
token ĐẦU TIÊN của chính câu trả lời.

Với câu đóng, "Yes"/"No" thường nằm đúng ở phần bị cắt. Arm A0-bug (`--raw-decode`) chạy với adapter gốc.

🟨 Đây là lỗi **phụ thuộc phiên bản**: code eval gốc của LLaVA cũng cắt như vậy, và từng đúng ở thời
`generate()` trả về prompt+đáp án. Không phải tác giả AMANDA cẩu thả — nhưng hệ quả là **code phát
hành không tái lập được số trong bài trên môi trường hiện tại**.

## Bản vá
Chỉ cắt khi `output_ids` **thật sự** mở đầu bằng `input_ids`. Nếu không, giữ nguyên toàn bộ.
"""
from __future__ import annotations

import torch

from amanda.mllm.adapters.llava_med_adapter import LLaVAMedAdapter


class PatchedLLaVAMedAdapter(LLaVAMedAdapter):
    """Giống hệt bản gốc, trừ `_decode_output` biết kiểm trước khi cắt."""

    def _decode_output(self, output_ids: torch.Tensor, input_ids: torch.Tensor, conv) -> str:
        n_in = input_ids.shape[1]
        echoed = (
            output_ids.shape[1] > n_in
            and torch.equal(output_ids[0, :n_in].cpu(), input_ids[0].cpu())
        )
        token_ids = output_ids[0, n_in:] if echoed else output_ids[0]

        output = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        for prefix in ("###", "Assistant:", "Response:"):
            if output.startswith(prefix):
                output = output[len(prefix):].strip()
        for stop_str in (conv.sep, getattr(conv, "sep2", None)):
            if stop_str and stop_str in output:
                output = output.split(stop_str, 1)[0].strip()
        return output
