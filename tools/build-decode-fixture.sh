#!/bin/sh
# Regenerates the decoder fixture: assembles tests/fixtures/hello.s, extracts .text, and records the listing objdump
# produces. The listing is the expectation the test compares against, so it must come from binutils, not from the decoder
# agreeing with itself. Requires binutils; the committed outputs mean tests do not.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
as -o "$work/hello.o" "$root/tests/fixtures/hello.s"
objcopy -O binary -j .text "$work/hello.o" "$root/tests/fixtures/hello.text.bin"
objdump -d --no-show-raw-insn "$work/hello.o" | awk '
  /^ *[0-9a-f]+:\t/ {
    split($1, offset, ":"); op = $2;
    printf "%s %s %s\n", offset[1], op, $3
  }' > "$work/listing.txt"
node - "$work/listing.txt" "$root/tests/fixtures/hello.dis.json" <<'JS'
const fs = require("node:fs");
const lines = fs.readFileSync(process.argv[2], "utf8").trim().split("\n");
const entries = lines.map((line) => {
  const [offset, mnemonic] = line.split(" ");
  return { offset: parseInt(offset, 16), mnemonic };
});
for (let index = 0; index < entries.length; index += 1) {
  entries[index].length = (entries[index + 1]?.offset ?? fs.statSync(process.argv[3].replace("hello.dis.json", "hello.text.bin")).size) - entries[index].offset;
}
fs.writeFileSync(process.argv[3], JSON.stringify(entries, null, 2) + "\n");
JS
echo "fixture rebuilt: $(wc -c < "$root/tests/fixtures/hello.text.bin") bytes, $(node -e "console.log(require('$root/tests/fixtures/hello.dis.json').length)") instructions"
