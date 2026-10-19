# Nguồn và quyền sử dụng của question bank

`qbank_v1.yaml` là một đồ thị câu hỏi đọc phim X-quang, CT và MRI, dịch sang tiếng Anh để dùng cho arm FQ. Các câu
hỏi được dựng từ **bài giảng và bảng kiểm (checklist) đọc phim chẩn đoán hình ảnh bằng tiếng Việt**. Mỗi câu hỏi giữ
**trích dẫn nguyên văn tiếng Việt** từ tài liệu nguồn (trường `vi_src`) kèm mã của đoạn trích (trường `src`).

- **Tác giả và cơ sở** của từng tài liệu được liệt kê trong khối `sources` ở đầu `qbank_v1.yaml` (tên tài liệu, tác
  giả, đơn vị). Mã `src` có dạng `<nguồn>:<mã tài liệu>#<số thứ tự đoạn>`; phần `<nguồn>:<mã tài liệu>` trỏ về
  một mục trong khối đó.
- Các trích dẫn được dùng cho **mục đích học thuật, phi thương mại**: một seminar môn học và gói tái lập thực nghiệm
  đi kèm. Bản quyền nội dung trích dẫn thuộc về tác giả và đơn vị gốc.
- Giấy phép MIT ở thư mục gốc của repo áp dụng cho **mã nguồn** của gói, không áp dụng cho nội dung trích dẫn trong
  `qbank_v1.yaml`.
- Tác giả hoặc đơn vị nào muốn gỡ hay sửa phần trích dẫn tài liệu của mình, xin liên hệ chủ repo (mở issue trên
  GitHub hoặc liên hệ trực tiếp); phần đó sẽ được gỡ.

`qbank_v1.yaml` được đóng băng bằng sha256 trong `data/qbank/FROZEN`. Bước 11 và `run_pipeline.py --arm FQ` kiểm mã
này trước khi chạy, nên mọi chỉnh sửa (kể cả phần chú thích) đều cần một phiên bản bank mới.
