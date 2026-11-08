# MTP-Healing cho model post-trained từ Nemotron 3.5 Lightning

## Kết quả kiểm tra bộ `magic_code`

Bộ được cung cấp có **đúng 4 file Python** (không tính thư mục cache):

| File cũ | Vai trò xác minh được trong nội dung |
| --- | --- |
| `recipes/nemotronh/__init__.py` | Export `nemotron_3_super_mtp_healing_config`. |
| `recipes/nemotronh/nemotron_3_super.py` | Recipe healing; detach MTP; pre-wrap hook chỉ bật gradient cho `mtp.layers.*`; khóa cập nhật expert bias của router backbone. |
| `data/datasets/sft.py` | Truyền `last_assistant_only_loss` từ factory vào chat dataset. |
| `data/datasets/utils.py` | Tạo mask cho span assistant cuối; báo lỗi nếu HF chat template thiếu `{% generation %}`. |

Đây là xác nhận **nội dung bộ file**, không phải xác nhận rằng lần train cũ chỉ có 4 thay đổi trên server. Thư mục `/opt/Megatron-Bridge` không tồn tại trên máy kiểm tra; không có Git diff của server, lệnh launch, config override hay snapshot tokenizer của lần chạy cũ. Hai file dataset thuộc đường chat HF; recipe healing cũ mặc định lại dùng `input/output`, nên thay đổi mask đó chỉ có hiệu lực nếu launcher đổi dataset sang chat.

Các phụ thuộc có thể nằm ngoài 4 file: phiên bản Megatron-Core hỗ trợ `mtp_detach_heads`, `frozen_expert_bias` trong gradient finalization, checkpoint còn MTP, tokenizer/chat template và launcher/config dữ liệu. Nếu template thiếu generation block, bộ code cũ yêu cầu sửa template ở ngoài bốn file. Không đủ bằng chứng để khẳng định bạn đã sửa những file đó trong lần chạy cũ.

## Cơ sở và phiên bản

Trang 26, §3.2 của **NVIDIA Nemotron 3 Super Technical Report** mô tả: freeze các trọng số ngoài MTP, dùng lại prompt RLVR, train MTP bằng negative log likelihood trên response được generate. Phần này không cung cấp LR, số step, batch size hay chiều dài context để tái lập chính xác. Code ở đây áp dụng cùng phương pháp cho Lightning, không khẳng định tái lập mọi hyperparameter của NVIDIA.

Nguồn kiến trúc: [HF config Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/main/config.json), [tài liệu Bridge Lightning](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/nemotron/nemotron3.5-lightning.html).

Code viết cho checkout Bridge `ec4ec358c`, Megatron-Core `feba4e1eb31bdc217e9e3f767baaddf3fdd17854`. Dùng container CUDA tương thích với checkout này và B200. Đây **không phải** patch drop-in cho mọi phiên bản Bridge cũ: bản hiện tại đã thay `sft.py` bằng cấu trúc dataset/config mới. Đặc biệt cần có:

- `model.freeze_base_model_for_mtp`, hook `_freeze_base_model_for_mtp` trong `training/setup.py`, và Core hỗ trợ chạy backbone với `no_grad`.
- `mtp_detach_heads` và xử lý `frozen_expert_bias` trong Core.
- `DirectHFSFTDatasetConfig`, `ChatSFTPreprocessingConfig(loss_mode="last_turn")` và logic chat Lightning hiện có.
- Nạp HF local qua `checkpoint.pretrained_checkpoint`, kiểm tra các weight mapping bắt buộc.

Không chép đè hai file dataset cũ vào checkout mới. Các sửa đổi có sẵn về skip nonfinite ở `training/config.py`, `training/train.py`, `training/utils/train_utils.py`, `tests/unit_tests/training/test_train.py` được giữ nguyên; recipe này không phụ thuộc vào bật cờ đó.

## Các file mới và hành vi

- `src/megatron/bridge/recipes/nemotronh/nemotron_3_5_lightning.py`: `nemotron_3_5_lightning_mtp_healing_config`.
- `src/megatron/bridge/recipes/nemotronh/__init__.py`: export recipe.
- `examples/models/nemotron/nemotron_3/lightning/mtp_healing.py`: entry point cho HF/Megatron và hai dạng dữ liệu.
- Hai file test ở `tests/unit_tests/examples/test_lightning_mtp_healing.py` và `tests/unit_tests/recipes/nemotronh/test_nemotron_3_5_lightning_healing.py`.

Model metadata được đọc từ `--hf-model-path` của **model bạn**, không lấy weight từ model NVIDIA rồi ghi đè checkpoint bạn. Với HF checkpoint, mặc định đường này là `--pretrained-checkpoint`. Với Megatron checkpoint, truyền HF snapshot tương ứng; cần có `config.json` và các file tokenizer.

Chỉ `mtp.layers.*` có gradient. Embedding, LM output head, backbone và cập nhật expert bias của router backbone bị khóa bằng cơ chế native. `mtp_detach_heads=True` tách các đầu vào/head dùng chung khỏi graph MTP. Callback trước step đầu kiểm tra trực tiếp `requires_grad`, sự tồn tại của tham số trainable và cờ khóa router trên từng rank; callback này không thay thế việc so sánh giá trị tensor trước/sau train.

Lightning có **một block MTP chung trọng số**, bên trong là attention + MoE (`*E`). `mtp_num_layers=2` là hai độ sâu dự đoán qua block lặp; HF thường ghi `num_nextn_predict_layers=1`. Không tạo hai bộ trọng số MTP độc lập.

Thiết lập khởi đầu: BF16, TP=1, PP=1, CP=1, EP=16, ETP=1, micro batch=1, global batch=128, context=4096, LR=5e-6, warmup=50, train=1000 step. LR/optimizer/schedule kế thừa `_sft_common`; MTP coefficient=0.3 như recipe Lightning SFT và bộ Super cũ. Vẫn giữ MoE auxiliary balancing loss mặc định từ Bridge (`1e-4`) và weight decay; đây không phải một objective loại bỏ toàn bộ regularization. Nếu muốn thí nghiệm NLL thuần, sửa `cfg.model.moe_aux_loss_coeff` và weight decay/schedule tương ứng rồi ghi nhận đó là thí nghiệm khác.

Dùng all-to-all, không cần DeepEP; packing và CUDA graphs tắt. Đây là cấu hình khởi đầu, **chưa benchmark hay xác nhận fit/throughput trên B200**. 4096 là ví dụ, không phải độ dài bắt buộc của healing. Với rollout dài cần chọn context đủ dài: truncation có thể làm mất response được giám sát. Kiểm tra sample đã tokenize và số token có loss trước khi chạy dài.

## Chuẩn bị dữ liệu

Response nên là rollout từ checkpoint post-train bạn muốn heal. Dùng prompt RLVR/mixture gần workload mục tiêu; tách held-out prompts để đo chất lượng MTP. Script không tự generate rollout.

### `messages` (khuyến nghị)

Mỗi dòng là một JSON object, kết thúc bằng assistant response đã generate:

```json
{"messages":[{"role":"user","content":"Tính 1 + 1."},{"role":"assistant","reasoning_content":"Cộng hai số một.","content":"2"}],"chat_template_kwargs":{"enable_thinking":true,"truncate_history_thinking":true}}
```

Giữ `reasoning_content`, top-level `tools`, assistant `tool_calls` nếu rollout có dùng. Dùng tokenizer/chat template chính thức của snapshot; không thay bằng ChatML generic. `chat_template_kwargs` phải khớp cách render prompt lúc lấy rollout. Đặc biệt, chỉ giữ reasoning lịch sử nếu rollout cũng giữ nó.

Mặc định `--loss-mode last_turn` chỉ giám sát assistant turn cuối, bao gồm reasoning/content/tool calls được mask đúng bởi Bridge. Dùng `--loss-mode assistant` nếu muốn giám sát tất cả assistant turns trong trajectory. Với trajectory có tool, thường lưu từng prefix + assistant response làm một row cho `last_turn`, hoặc dùng `assistant` cho toàn trajectory. Không dùng full-sequence loss vì sẽ train cả prompt/user/tool response.

### `input/output`

`input` phải là **prompt đã render hoàn chỉnh đúng lúc generation**; `output` là phần continuation model đã sinh. Nếu hiện tại `input` chỉ là câu hỏi tự nhiên chưa qua template, chuyển sang `messages` hoặc render trước.

```json
{"input":"<|im_start|>user\nTính 1 + 1.<|im_end|>\n<|im_start|>assistant\n<think>\n","output":"Cộng hai số một.\n</think>\n2<|im_end|>"}
```

Ví dụ minh họa schema; hãy dùng `apply_chat_template` của snapshot để tạo prefix thật. Script không chèn dấu cách, không strip whitespace, không tự thêm BOS/EOS, và chỉ tính loss trên `output`. Vì vậy cần giữ token kết thúc trong continuation nếu muốn supervise nó. Hai trường được tokenize riêng rồi nối; với serialization đặc biệt hãy kiểm tra token IDs có khớp rollout gốc, nhất là ranh giới prefix/continuation.

## Chạy trên 2 nodes B200

Ví dụ giả định **8 GPU/node** và shared filesystem. Chạy từ repo root trong cùng container/môi trường trên cả hai node, cùng code/version, paths và config. Với allocation khác, chỉnh `--nproc_per_node` và EP; cả `WORLD_SIZE % (TP*CP)` và `WORLD_SIZE % EP` phải bằng 0, EP chia hết 128 experts. Global batch phải chia hết `DP * micro_batch`; với 16 GPU và TP=CP=1, gradient accumulation là 8.

Schema preflight (không cần CUDA, không kiểm tra tensor weights/tokenization):

```bash
uv run --no-sync python examples/models/nemotron/nemotron_3/lightning/mtp_healing.py \
  --pretrained-checkpoint /checkpoints/my-lightning-posttrain-hf \
  --train-data /data/healing/train.jsonl \
  --output-dir /runs/lightning-mtp-healing \
  --data-format messages --check-inputs
```

Trên **cả hai node**, đặt `MASTER_ADDR` thành địa chỉ có thể truy cập của node 0; `NODE_RANK=0` trên node 0, `NODE_RANK=1` trên node 1. Tránh trùng port với job khác.

```bash
export MASTER_ADDR=NODE0_REACHABLE_ADDRESS
export MASTER_PORT=29500
export NODE_RANK=0  # node còn lại dùng 1
export PYTHONPATH="${PWD}/src:${PWD}/3rdparty/Megatron-LM:${PYTHONPATH:-}"

uv run --no-sync python -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  examples/models/nemotron/nemotron_3/lightning/mtp_healing.py \
  --pretrained-checkpoint /checkpoints/my-lightning-posttrain-hf \
  --train-data /data/healing/train.jsonl \
  --output-dir /runs/lightning-mtp-healing \
  --data-format messages --loss-mode last_turn \
  --tp 1 --ep 16 --cp 1 --seq-length 4096 \
  --global-batch-size 128 --micro-batch-size 1 \
  --lr 5e-6 --train-iters 1000 --warmup-iters 50 --save-interval 100
```

Nạp **Megatron**: thay/đặt hai option dưới đây, dùng checkpoint chứa model weights đầy đủ và còn MTP:

```bash
--pretrained-checkpoint /checkpoints/my-lightning-posttrain-megatron \
--hf-model-path /checkpoints/my-lightning-posttrain-hf
```

Chọn `--data-format input-output` cho dữ liệu đã render. Có thể truyền nhiều JSONL sau `--train-data`. Lần đầu dùng output directory mới/trống. **Resume healing** bằng cùng command, cùng data/config và thêm `--resume`; optimizer/scheduler/RNG được phục hồi từ `output-dir/checkpoints`. Không dùng `--resume` để bắt đầu từ optimizer của lần SFT/RL cũ.

Nếu post-training đã bỏ MTP weights/config, khôi phục một MTP block tương thích trước khi healing. Không chỉ đổi `num_nextn_predict_layers` cho qua check: loader vẫn cần trọng số MTP thực tế. Recipe từ chối metadata thiếu MTP; HF loader kiểm tra mappings, native loader dùng `raise_unexpected` để lỗi khi thiếu tensor cần nạp.

## Kiểm chứng trước khi train dài và export

Chạy thử 5 step trên cluster với output directory riêng, `--train-iters 5 --warmup-iters 1 --save-interval 5`, giữ nguyên các option còn lại. Xác nhận log `MTP freeze verified`, `mtp_1 loss`, `mtp_2 loss`, gradient hữu hạn, checkpoint được lưu và resume thành công. `lm loss` vẫn có thể được log dù backbone bị freeze; không dùng riêng đường LM loss để đánh giá MTP healing.

Để xác minh freeze đầy đủ, export snapshot trước/sau theo cùng cách rồi so sánh các tensor ngoài `mtp.*`, gồm embedding, LM head và router expert biases; chúng phải giữ nguyên. MTP weights cần thay đổi sau optimizer step. Đo acceptance rate/tokens accepted của MTP trên held-out rollout và workload serving thực tế; NLL giảm chưa đủ chứng minh tăng tốc decoding.

Checkpoint đầu ra là native Megatron, lưu cả model. Export HF bằng công cụ sẵn có, chạy trong môi trường GPU tương thích (ví dụ layout 1 node EP=8 để reshard):

```bash
./scripts/conversion/convert.sh export \
  --executor local --device gpu --gpus-per-node 8 \
  --tp 1 --pp 1 --ep 8 --etp 1 \
  --hf-model /checkpoints/my-lightning-posttrain-hf \
  --megatron-path /runs/lightning-mtp-healing/checkpoints/iter_0001000 \
  --hf-path /exports/my-lightning-healed-hf \
  --torch-dtype bfloat16 --distributed-save
```

Kiểm tra export còn `mtp.*` và dùng đúng config/tokenizer. Không tái sử dụng output prefix của input checkpoint.

## Tests và giới hạn xác minh

Đã chạy local: 26 test CPU về schema/paths, resume, topology/batch và runtime freeze guard; Ruff; kiểm tra cú pháp. Các test CPU dùng entry point thật với tham số/tensor metadata giả lập cho freeze guard, không chạy autograd của model thật.

```bash
uv run --no-project --with pytest python -m pytest --noconftest -o addopts='' \
  tests/unit_tests/examples/test_lightning_mtp_healing.py -q
```

Test recipe cần môi trường Bridge đầy đủ:

```bash
uv run --no-sync python -m pytest \
  tests/unit_tests/recipes/nemotronh/test_nemotron_3_5_lightning_healing.py \
  tests/unit_tests/models/nemotronh/test_nemotron_3_5_chat.py -q
```

Chưa chạy được test recipe với stack Megatron thật, nạp checkpoint thực tế, forward/backward, export hay benchmark B200 ở máy local vì môi trường này thiếu PyTorch/CUDA. Không xem kết quả CPU là chứng nhận training end-to-end.
