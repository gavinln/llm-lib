let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd /mnt/d/ws/llm-lib/openai-examples
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +127 long-content-extraction.py
badd +49 ~/.cache/pypoetry/virtualenvs/openai-examples-AJjK4Rvk-py3.11/lib/python3.11/site-packages/openai/resources/chat/completions.py
badd +44 ~/.cache/pypoetry/virtualenvs/openai-examples-AJjK4Rvk-py3.11/lib/python3.11/site-packages/openai/types/chat/chat_completion.py
badd +1 /mnt/d/ws/windows-ubuntu-tips/recent-changes.md
badd +15 ~/.cache/pypoetry/virtualenvs/openai-examples-AJjK4Rvk-py3.11/lib/python3.11/site-packages/openai/types/chat/chat_completion_message_param.py
badd +25 ~/.cache/pypoetry/virtualenvs/openai-examples-AJjK4Rvk-py3.11/lib/python3.11/site-packages/openai/types/chat/chat_completion_message.py
badd +77 ~/.cache/pypoetry/virtualenvs/openai-examples-AJjK4Rvk-py3.11/lib/python3.11/site-packages/openai/_models.py
badd +14 /mnt/d/ws/obsidian-vault/notes/dailies/2024-05-16.md
badd +75 /mnt/d/ws/llm-lib/redis-examples/vector-search-existing.py
argglobal
%argdel
$argadd long-content-extraction.py
edit long-content-extraction.py
argglobal
balt /mnt/d/ws/obsidian-vault/notes/dailies/2024-05-16.md
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 107 - ((27 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 107
normal! 028|
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
let g:this_session = v:this_session
let g:this_obsession = v:this_session
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
