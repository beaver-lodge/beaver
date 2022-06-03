set -e
rm -rf doc
mix docs
open doc/index.html
