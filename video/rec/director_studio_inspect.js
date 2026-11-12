// Clip 2: studio_inspect.mp4. A finished ticket_router thread (threadId in the URL): hover steps (the graph lights the node that ran), expand
// the triage step and its value to see what the node wrote into the state, then open a checkpoint's "View state" (Values / JSON).
// Needs: tlsproxy.py 2128 2125, rec --css 1440x810 --pre-js video/rec/studio_prejs.js, URL ...&threadId=<finished thread>#panes=56
const chevron = (name) => { const t = studioStep(name); return t && t.closest('div').parentElement.querySelector('button[aria-label="Expand node"],button[aria-label="Collapse node"]'); };
// value chip / row `name` inside the expanded step `stepName` (Studio renders them as div[role=button])
const chipOf = (stepName, name, depthPx) => {
  const top = studioStep(stepName).getBoundingClientRect().top;
  return qa('button,[role=button]').filter(b => isVisible(b) && b.getBoundingClientRect().left > innerWidth * 0.5 && b.getBoundingClientRect().top > top && b.getBoundingClientRect().top < top + (depthPx || 400) && norm(b.textContent).toLowerCase().startsWith(name))[0] || null;
};

await waitFor(() => q('.react-flow__node') && studioStep('escalate'), 40000, 'graph + steps');
await sleep(500);
await click('css:button[aria-label="Collapse input editor"]', { timeout: 4000 });
await sleep(300);
await studioFocus(['__start__', 'triage', 'escalate'], { ms: 900 });
rec.mark('studio_loaded');
await sleep(500);

// ---- hover the steps: the graph lights the node that ran in that step
for (const name of ['load_ticket', 'triage']) {
  await studioHover(studioStep(name));
  rec.mark('hover_' + name);
  await sleep(900);
}

// ---- expand triage: the state the step wrote (category, priority, ... and why it escalated)
await click(chevron('triage'));
rec.mark('triage_expanded');
await sleep(700);
await click(chipOf('triage', 'triage', 150));                          // the structured classifier output
rec.mark('triage_value_open');
await sleep(2200);
await safe('open lexicon_hits', async () => {
  await click(chipOf('triage', 'lexicon_hits', 600));
  rec.mark('lexicon_hits_open');
  await sleep(2000);
});

// ---- collapse it again, then open the checkpoint after triage: "View state" = the thread state at that point (Values / JSON)
await click(chevron('triage'));
await sleep(500);
const escTop = studioStep('escalate').getBoundingClientRect().top;
const cp = studioCheckpoints().filter(g => g.getBoundingClientRect().top < escTop).pop();
fakeHover(cp);
await sleep(500);
await click([...cp.querySelectorAll('button')].find(b => /View state/.test(b.textContent)));
rec.mark('view_state_open');
await sleep(2800);
await click(byText('JSON', { exact: true }));
rec.mark('view_state_json');
await sleep(1500);
await safe('scroll JSON', async () => {                                 // the popover is scrollable: bring the triage block + `next` into view
  const lab = byText('Viewing checkpoint');
  let sc = lab; while (sc && !(/(auto|scroll)/.test(getComputedStyle(sc).overflowY) && sc.scrollHeight > sc.clientHeight + 20)) sc = sc.parentElement;
  if (!sc) sc = qa('*').filter(e => /(auto|scroll)/.test(getComputedStyle(e).overflowY) && e.scrollHeight > e.clientHeight + 20 && isVisible(e) && lab && e.contains(byText('values:', { exact: false })))[0];
  rec.log('popover scroller: ' + (sc ? sc.tagName + ' ' + sc.scrollHeight + '/' + sc.clientHeight : 'none'));
  if (sc) await smoothScrollBy(sc, 260, 1500);
  rec.mark('view_state_json_scrolled');
});
await sleep(1800);
rec.mark('end');
