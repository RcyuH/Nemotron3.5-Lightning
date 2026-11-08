# Nemotron 3.5 Lightning — MTP Healing

Snapshot mã nguồn để train MTP healing cho model post-trained từ Nemotron 3.5 Lightning 30B-A3B.

- `training.py`: entry point offline, cấu hình theo file training Super đã được chuyển sang Lightning; ví dụ 2 node × 8 B200.
- `Megatron-Bridge/`: toàn bộ mã nguồn Bridge, bao gồm recipe Lightning, hướng dẫn và các thay đổi training loop local.
- `Nemotron/`: toàn bộ mã nguồn Nemotron, kèm các script xử lý dữ liệu và recipe CPT local.
- `mtp-healing-lightning.patch`: patch của phần healing đã tạo trước `training.py`; không đại diện cho tất cả thay đổi trong snapshot này.
- `SOURCE_SNAPSHOT.json`: upstream URL và commit gốc của từng repo.

## Clone

```bash
git clone --recurse-submodules https://github.com/RcyuH/Nemotron3.5-Lightning.git
cd Nemotron3.5-Lightning
```

Megatron-Core được giữ dưới dạng submodule, pin đúng commit của workspace. Nếu đã clone không có submodule:

```bash
git submodule update --init --recursive
```

## Training

Dùng môi trường CUDA/NeMo đã chuẩn bị, cài Bridge trong `Megatron-Bridge/` và dùng Core đi kèm. `training.py` cần các API native của checkout này.

```bash
python training.py --help
```

Xem [hướng dẫn healing](Megatron-Bridge/examples/models/nemotron/nemotron_3/lightning/MTP_HEALING.vi.md) để biết định dạng dữ liệu và cơ chế freeze. Hướng dẫn đó mô tả entry point riêng `mtp_healing.py`; `training.py` giữ mặc định theo file mẫu: 8K, 7 bước MTP lặp, loss weight 1.0, LR 1e-5, GBS 64, không packing, 2 epoch-equivalents.

Checkpoint và dữ liệu training không nằm trong repository. Code đã được kiểm tra CPU/lint cục bộ; chưa xác nhận training end-to-end trên B200.

## Nguồn và giấy phép

Đây là snapshot nội dung working tree, không nhập toàn bộ Git history upstream. Các LICENSE và thông báo bản quyền upstream được giữ nguyên trong từng thư mục. Repo gồm cả thay đổi local đã có trước phần MTP healing.
