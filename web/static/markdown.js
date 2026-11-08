/* Small, safe Markdown renderer for ZENIT replies.
   All text is HTML-escaped before any markup is added, so model output can
   never inject tags or scripts. Supports headings, lists (nested), quotes,
   tables, fenced code with light highlighting, inline code, emphasis,
   links (http/https only) and source citations like [S1]. */
(function () {
  'use strict';
  const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ESC[c]);

  function inline(src) {
    const codes = [];
    let s = src.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return '\u0000' + (codes.length - 1) + '\u0000'; });
    s = esc(s);
    s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, '$1<em>$2</em>');
    s = s.replace(/(^|[^_\w])_([^_\n]+)_(?!\w)/g, '$1<em>$2</em>');
    s = s.replace(/~~([^~\n]+)~~/g, '<del>$1</del>');
    s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    s = s.replace(/\[(S\d{1,2})\]/g, '<span class="cite" title="Source $1 from your documents">$1</span>');
    s = s.replace(/\u0000(\d+)\u0000/g, (_, i) => '<code>' + esc(codes[+i]) + '</code>');
    return s;
  }

  const KEYWORDS = new Set(('def return if elif else for while in not and or is import from as class try except finally ' +
    'raise with lambda yield pass break continue global None True False function const let var new this async await ' +
    'export default switch case typeof null undefined true false public private protected static void int float double ' +
    'string bool fn mut struct impl enum match local then end nil package func go defer interface select where using namespace').split(' '));
  const HASH_LANGS = new Set(['python', 'py', 'sh', 'bash', 'shell', 'zsh', 'ruby', 'rb', 'yaml', 'yml', 'toml', 'r', 'powershell', 'ps1']);
  const DASH_LANGS = new Set(['lua', 'sql', 'luau']);

  function highlight(code, lang) {
    lang = (lang || '').toLowerCase();
    const comment = HASH_LANGS.has(lang) ? '#[^\\n]*' : DASH_LANGS.has(lang) ? '--[^\\n]*' : '\\/\\/[^\\n]*|#[^\\n]*';
    const re = new RegExp('(' + comment + ')|("(?:[^"\\\\\\n]|\\\\.)*"|\'(?:[^\'\\\\\\n]|\\\\.)*\'|`(?:[^`\\\\]|\\\\.)*`)|\\b(\\d+(?:\\.\\d+)?)\\b|\\b([A-Za-z_]\\w*)\\b', 'g');
    let out = '', last = 0, m;
    while ((m = re.exec(code))) {
      out += esc(code.slice(last, m.index));
      if (m[1]) out += '<span class="tok-c">' + esc(m[1]) + '</span>';
      else if (m[2]) out += '<span class="tok-s">' + esc(m[2]) + '</span>';
      else if (m[3]) out += '<span class="tok-n">' + m[3] + '</span>';
      else if (m[4]) out += KEYWORDS.has(m[4]) ? '<span class="tok-k">' + m[4] + '</span>' : m[4];
      last = re.lastIndex;
    }
    return out + esc(code.slice(last));
  }

  function codeBlock(code, lang) {
    const label = lang ? esc(lang) : 'code';
    return '<div class="codeblock"><div class="code-head"><span>' + label + '</span>' +
      '<button type="button" data-copy-code>Copy code</button></div><pre><code>' +
      highlight(code, lang) + '</code></pre></div>';
  }

  const RE = {
    fence: /^\s*```\s*([\w+#.-]*)\s*$/,
    heading: /^(#{1,4})\s+(.*)$/,
    hr: /^\s*(-{3,}|\*{3,}|_{3,})\s*$/,
    quote: /^\s*>\s?(.*)$/,
    ul: /^(\s*)[-*+]\s+(.*)$/,
    ol: /^(\s*)(\d+)[.)]\s+(.*)$/,
    tableSep: /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/,
  };

  function list(lines, start) {
    // Builds nested lists from indentation. Returns [html, nextIndex].
    const items = [];
    let i = start;
    while (i < lines.length) {
      const m = lines[i].match(RE.ul) || lines[i].match(RE.ol);
      if (!m) {
        if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && items.length) {
          items[items.length - 1].text += ' ' + lines[i].trim();
          i++;
          continue;
        }
        break;
      }
      const ordered = RE.ol.test(lines[i]);
      const indent = m[1].replace(/\t/g, '    ').length;
      const text = ordered ? m[3] : m[2];
      items.push({ indent, ordered, text, num: ordered ? +m[2] : null });
      i++;
    }
    function build(arr, from, level) {
      let html = '', k = from;
      const ordered = arr[from].ordered;
      const startNum = arr[from].num;
      html += ordered ? (startNum && startNum !== 1 ? '<ol start="' + startNum + '">' : '<ol>') : '<ul>';
      while (k < arr.length && arr[k].indent >= level) {
        if (arr[k].indent > level) {
          const [inner, next] = build(arr, k, arr[k].indent);
          html = html.replace(/<\/li>$/, '') + inner + '</li>';
          k = next;
          continue;
        }
        let t = arr[k].text;
        let box = '';
        const task = t.match(/^\[( |x|X)\]\s+(.*)$/);
        if (task) { box = task[1] === ' ' ? '☐ ' : '☑ '; t = task[2]; }
        html += '<li>' + box + inline(t) + '</li>';
        k++;
      }
      return [html + (ordered ? '</ol>' : '</ul>'), k];
    }
    if (!items.length) return ['', start + 1];
    let html = '', k = 0;
    while (k < items.length) {
      const [h, next] = build(items, k, items[k].indent);
      html += h;
      k = next;
    }
    return [html, i];
  }

  function splitRow(line) {
    let s = line.trim();
    if (s.startsWith('|')) s = s.slice(1);
    if (s.endsWith('|')) s = s.slice(0, -1);
    return s.split('|').map((c) => c.trim());
  }

  function render(text) {
    const lines = String(text || '').replace(/\r\n?/g, '\n').split('\n');
    let html = '', para = [], i = 0;
    const flush = () => { if (para.length) { html += '<p>' + para.map(inline).join('<br>') + '</p>'; para = []; } };
    while (i < lines.length) {
      const line = lines[i];
      let m;
      if ((m = line.match(RE.fence))) {
        flush();
        const lang = m[1];
        const body = [];
        i++;
        while (i < lines.length && !RE.fence.test(lines[i])) body.push(lines[i++]);
        i++;
        html += codeBlock(body.join('\n'), lang);
        continue;
      }
      if (!line.trim()) { flush(); i++; continue; }
      if ((m = line.match(RE.heading))) { flush(); const n = m[1].length; html += '<h' + n + '>' + inline(m[2]) + '</h' + n + '>'; i++; continue; }
      if (RE.hr.test(line)) { flush(); html += '<hr>'; i++; continue; }
      if (RE.quote.test(line)) {
        flush();
        const q = [];
        while (i < lines.length && RE.quote.test(lines[i])) q.push(lines[i++].match(RE.quote)[1]);
        html += '<blockquote>' + render(q.join('\n')) + '</blockquote>';
        continue;
      }
      if (line.includes('|') && i + 1 < lines.length && RE.tableSep.test(lines[i + 1])) {
        flush();
        const head = splitRow(line);
        i += 2;
        let t = '<table><thead><tr>' + head.map((h) => '<th>' + inline(h) + '</th>').join('') + '</tr></thead><tbody>';
        while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
          t += '<tr>' + splitRow(lines[i]).map((c) => '<td>' + inline(c) + '</td>').join('') + '</tr>';
          i++;
        }
        html += t + '</tbody></table>';
        continue;
      }
      if (RE.ul.test(line) || RE.ol.test(line)) {
        flush();
        const [h, next] = list(lines, i);
        html += h;
        i = next;
        continue;
      }
      para.push(line);
      i++;
    }
    flush();
    return html;
  }

  window.ZenitMD = { render, escape: esc };
})();
