#!/usr/bin/env bash
# Assemble the single-file builds of Mesh Quest from game/src + game/js/engine.js.
#
#   ./game/build.sh [artifact-output-path]
#
# Outputs:
#   game/mesh-quest.html  standalone page (open locally, email, self-host)
#   docs/play.html       the same page plus analytics, served by GitHub Pages
#   $1 (optional)         artifact variant: no document skeleton, no analytics
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/game/src"
engine="$root/game/js/engine.js"
untangle="$root/game/js/untangle.js"
arcs="$root/game/js/arcs.js"
domains="$root/game/js/curved_domains.js"
gif="$root/game/js/gif.js"

# Analytics live only on the public Pages build: the artifact runtime blocks
# external requests, and the local standalone file should not phone home.
analytics='<script data-goatcounter="https://arjunnarayanan.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>'

# Social-card metadata for link previews (LinkedIn, Slack, iMessage, X...).
# Absolute URLs are required: preview crawlers do not resolve relative paths.
site="https://arjunnarayanan.github.io/par-quad/"
social='<meta name="description" content="Slice shapes into perfect four-sided blocks. Simple to play, tricky to master.">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Mesh Quest">
<meta property="og:title" content="Mesh Quest — slice shapes into blocks">
<meta property="og:description" content="A free puzzle game: cut each shape into four-sided blocks and chase a perfect score. Easy to pick up, surprisingly deep.">
<meta property="og:url" content="'"$site"'play.html">
<meta property="og:image" content="'"$site"'preview.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="Three shapes — an L, a star and a square ring — each cut into four-sided blocks.">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Mesh Quest — slice shapes into blocks">
<meta name="twitter:description" content="A free puzzle game: cut each shape into four-sided blocks and chase a perfect score.">
<meta name="twitter:image" content="'"$site"'preview.png">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>%F0%9F%94%B7</text></svg>">'

emit_head() {
  echo '<title>Mesh Quest</title>'
  echo '<style>'
  cat "$src/style.css"
  echo '</style>'
}

emit_body() {
  cat "$src/body.html"
  echo '<script>'
  cat "$untangle"
  cat "$arcs"
  cat "$domains"
  cat "$engine"
  cat "$gif"
  cat "$src/ui.js"
  echo '</script>'
}

standalone() {  # "$@" = extra head blocks, each emitted on its own line
  echo '<!doctype html>'
  echo '<html lang="en">'
  echo '<head>'
  echo '<meta charset="utf-8">'
  echo '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
  emit_head
  for block in "$@"; do
    [ -n "$block" ] && printf '%s\n' "$block"
  done
  echo '</head>'
  echo '<body>'
  emit_body
  echo '</body>'
  echo '</html>'
}

standalone "" > "$root/game/mesh-quest.html"
mkdir -p "$root/docs"
standalone "$social" "$analytics" > "$root/docs/play.html"

if [ $# -ge 1 ]; then
  { emit_head; emit_body; } > "$1"
  echo "built: game/mesh-quest.html, docs/play.html, $1"
else
  echo "built: game/mesh-quest.html, docs/play.html"
fi
