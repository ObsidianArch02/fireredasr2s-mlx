#!/bin/bash
# Generic resumable parallel chunked download from a mirror endpoint
# usage: chunked_dl.sh <url> <size> <sha256> <dest_file> [nchunk] [par]
set -u
URL="$1"; SIZE="$2"; WANT_SHA="$3"; DEST="$4"; NCHUNK="${5:-24}"; PAR="${6:-12}"
# absolutize DEST before cd
DEST="$(cd "$(dirname "$DEST")" && pwd)/$(basename "$DEST")"
CHUNK=$(( (SIZE + NCHUNK - 1) / NCHUNK ))
DIR="$(dirname "$DEST")"; BASE="$(basename "$DEST")"
mkdir -p "$DIR/chunks_$BASE"
cd "$DIR/chunks_$BASE" || exit 1

download_chunk() {
  local i=$1
  local start=$(( i * CHUNK ))
  local end=$(( start + CHUNK - 1 ))
  if [ $end -ge $SIZE ]; then end=$(( SIZE - 1 )); fi
  local want=$(( end - start + 1 ))
  local f="c_$(printf '%03d' "$i")"
  while true; do
    local have=0
    [ -f "$f" ] && have=$(stat -f%z "$f")
    if [ "$have" -ge "$want" ]; then return 0; fi
    local rs=$(( start + have ))
    curl -sfL --max-time 600 -r "$rs-$end" "$URL" >> "$f" || true
    local now=0
    [ -f "$f" ] && now=$(stat -f%z "$f")
    if [ "$now" -ge "$want" ]; then return 0; fi
    if [ "$now" -eq "$have" ]; then sleep 15; fi
  done
}
export -f download_chunk
export URL SIZE CHUNK

while true; do
  INCOMPLETE=0
  for i in $(seq 0 $(( NCHUNK - 1 ))); do
    f="c_$(printf '%03d' "$i")"
    start=$(( i * CHUNK )); end=$(( start + CHUNK - 1 ))
    [ $end -ge $SIZE ] && end=$(( SIZE - 1 ))
    want=$(( end - start + 1 ))
    have=0; [ -f "$f" ] && have=$(stat -f%z "$f")
    [ "$have" -lt "$want" ] && INCOMPLETE=$((INCOMPLETE+1))
  done
  [ "$INCOMPLETE" -eq 0 ] && break
  echo "round: $INCOMPLETE incomplete"
  seq 0 $(( NCHUNK - 1 )) | xargs -P "$PAR" -I{} bash -c 'download_chunk {}' || true
  sleep 5
done

cat $(ls c_* | sort) > "$DEST"
ACTUAL=$(stat -f%z "$DEST")
if [ "$ACTUAL" -ne "$SIZE" ]; then echo "SIZE MISMATCH $ACTUAL != $SIZE"; exit 1; fi
GOT=$(shasum -a 256 "$DEST" | awk '{print $1}')
if [ "$GOT" != "$WANT_SHA" ]; then echo "SHA MISMATCH $GOT"; exit 1; fi
echo "OK $DEST ($SIZE bytes, sha256 verified)"
rm -rf "$DIR/chunks_$BASE"
