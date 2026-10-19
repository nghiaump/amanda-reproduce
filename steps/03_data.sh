#!/usr/bin/env bash
# Bước 03 — dữ liệu SLAKE (tập test tiếng Anh: 1.061 câu, 96 ảnh) + knowledge graph của SLAKE.
# Tải có resume (curl -C -). Kết thúc bằng self-test: sai số câu / số ảnh / ảnh thiếu là dừng.
source "$(dirname "$0")/../lib/common.sh"
start_step 03_data
need_venv
mkdir -p data/slake

fetch() {   # <tên file>
  local f=data/slake/$1
  log "tải $1"
  curl -fL -sS --retry 5 --retry-delay 5 -C - -o "$f" "$SLAKE_URL/$1" || die "tải $1 thất bại (chạy lại bước này để tải tiếp)"
  log "  $(du -h "$f" | cut -f1)"
}
[ -f data/slake/test.json ] || fetch test.json
got=$(sha256sum data/slake/test.json | cut -d' ' -f1)
[ "$got" = "$SLAKE_TEST_SHA256" ] || die "test.json sha256 $got khác bản đã dùng ($SLAKE_TEST_SHA256)"
log "✓ test.json khớp sha256"

if [ ! -f data/slake/imgs/imgs/xmlab102/source.jpg ] && [ ! -f data/slake/imgs/xmlab102/source.jpg ]; then
  fetch imgs.zip
  log "giải nén imgs.zip"
  unzip -q -o data/slake/imgs.zip -d data/slake/imgs -x '__MACOSX/*' || die "giải nén thất bại (xoá data/slake/imgs.zip rồi chạy lại)"
fi

"$PY" scripts/prepare_slake.py --self-test || die "self-test dữ liệu đỏ"
finish_step
