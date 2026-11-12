// Clip 3: studio_timetravel.mp4. A finished ticket_router thread (threadId in the URL). Every step boundary is a checkpoint: hover one and
// Studio offers "View state" and "Re-run from here". Re-run from the checkpoint before `triage`: the thread forks ("Fork 2 of 2"), triage and
// escalate run again on the new branch, and the fork arrows switch back to the original branch (Fork 1 of 2) and forward again.
// Needs: tlsproxy.py 2128 2125, rec --css 1440x810 --pre-js video/rec/studio_prejs.js, URL ...&threadId=<a FRESH finished thread>#panes=56
// (the clip adds a fork to its thread, so re-recording needs a new thread: see video/rec/record_studio.sh)
const submitBtn = () => qa('button[aria-label="Submit"]').find(isVisible);
const cancelBtn = () => qa('button').find(b => isVisible(b) && /^cancel$/i.test(norm(b.textContent)));
const frameRoute = async () => { await studioFocus(['__start__', 'triage', 'escalate'], { ms: 900 }); };

await waitFor(() => q('.react-flow__node') && studioStep('escalate'), 40000, 'graph + steps');
await sleep(500);
await click('css:button[aria-label="Collapse input editor"]', { timeout: 4000 });
await sleep(300);
await frameRoute();
rec.mark('studio_loaded');
await sleep(500);

// ---- the steps of the finished run, then the checkpoint between load_ticket and triage
await studioHover(studioStep('triage'));
rec.mark('hover_triage');
await sleep(900);
const trTop = () => studioStep('triage').getBoundingClientRect().top;
const cpBeforeTriage = () => studioCheckpoints().filter(g => g.getBoundingClientRect().top < trTop()).pop();
const cp = cpBeforeTriage();
studioUnhover();
await hover(cp);
fakeHover(cp);
rec.mark('checkpoint_hover');                                            // "View state . Re-run from here" appear on the divider
await sleep(1700);

// ---- Re-run from here
await click(cp.querySelector('button[aria-label="Re-run from here"]'));
rec.mark('rerun_clicked');
await waitFor(() => cancelBtn(), 15000, 'run started');
await sleep(500);
await frameRoute();                                                      // Studio re-centres the graph when a run starts
rec.mark('rerun_running');
await waitFor(() => submitBtn() && !cancelBtn(), 60000, 'rerun finished');
rec.mark('rerun_finished');
await sleep(900);
await frameRoute();                                                      // Studio re-centres again when the run ends
await sleep(900);                                                        // "Fork 2 of 2" on the new branch, new triage + escalate steps

// ---- the fork arrows: back to the original branch (Fork 1 of 2), look at its steps, then forward to the new one again
await safe('fork flip', async () => {
  await click('css:button[aria-label="Previous fork"]', { timeout: 6000 });
  rec.mark('fork_1_of_2');
  await sleep(500);
  await studioHover(studioStep('triage'));
  await sleep(1000);
  await studioHover(studioStep('escalate'));
  await sleep(1100);
  studioUnhover();
  await click('css:button[aria-label="Next fork"]', { timeout: 6000 });
  rec.mark('fork_2_of_2');
  await sleep(500);
  await studioHover(studioStep('escalate'));
  await sleep(1300);
});
rec.mark('end');
