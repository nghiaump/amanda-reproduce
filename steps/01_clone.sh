#!/usr/bin/env bash
# Bước 01 — clone mã nguồn của tác giả AMANDA và LLaVA-Med, GHIM đúng commit đã dùng.
# Hai repo này KHÔNG bị sửa: mọi thay đổi nằm trong src/amanda_lab (kế thừa lớp, ghi đè thuộc tính).
source "$(dirname "$0")/../lib/common.sh"
start_step 01_clone

clone_pinned() {   # <url> <thư mục> <commit>
  local url=$1 dir=$2 commit=$3
  if [ ! -d "$dir/.git" ]; then
    log "clone $url → $dir"
    git clone -q "$url" "$dir" || die "clone $url thất bại"
  fi
  git -C "$dir" fetch -q origin 2>/dev/null || true
  git -C "$dir" -c advice.detachedHead=false checkout -q "$commit" || die "không checkout được $commit trong $dir"
  [ -z "$(git -C "$dir" status --porcelain)" ] || die "$dir có thay đổi cục bộ — repo bên thứ ba phải nguyên vẹn (git -C $dir checkout .)"
  log "✓ $dir @ $(git -C "$dir" rev-parse --short HEAD)"
}

mkdir -p third_party
clone_pinned "$AMANDA_GIT" third_party/AMANDA   "$AMANDA_COMMIT"
clone_pinned "$LLAVA_GIT"  third_party/LLaVA-Med "$LLAVA_COMMIT"
need_file third_party/AMANDA/src/amanda/pipeline/__init__.py "cấu trúc repo AMANDA không như mong đợi"
finish_step
