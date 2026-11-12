// Clip 1: studio_run.mp4. ticket_router in Studio's graph view: pick the "Ready-made case" (Scenario enum dropdown), Submit,
// watch the run execute (the thread pane fills with one step per node; the graph dims everything except the active node), then walk
// the finished steps so the graph lights each node in turn. The thread id is stored in the 'thread' marker (used by clip 2).
// Needs: tlsproxy.py 2128 2125, rec --css 1440x810 --pre-js video/rec/studio_prejs.js, URL .../studio/thread?...&assistantId=<ticket_router>
const fit = async () => { await click('css:button[aria-label="Fit graph to view"]', { timeout: 6000 }); };
const submitBtn = () => qa('button[aria-label="Submit"]').find(isVisible);
const frameRoute = async () => { await studioFocus(['triage', 'escalate'], { ms: 1000 }); };     // zoom so triage .. escalate fill the pane
const vp = (l) => rec.log('VP ' + l + ' ' + studioVP());

await waitFor(() => q('.react-flow__node'), 40000, 'graph nodes');
await sleep(700);
rec.mark('studio_loaded');
await sleep(1100);                                                       // Studio as it opens: graph on top, the empty Input form below

// ---- open the Scenario ("Ready-made case") field and its dropdown
await click('scenario');
rec.mark('scenario_row_open');
await sleep(500);
await click('css:button[role=combobox]');
rec.mark('dropdown_open');
await sleep(2200);                                                       // viewer reads the 10 options
await hover(byText(/T-2185 billing/));
await sleep(350);
await hover(byText(/T-2304 heating/));
await sleep(350);
await hover(byText(/T-2301 URGENT/));
await sleep(500);
await click(byText(/T-2301 URGENT/));
rec.mark('scenario_picked');
await sleep(1100);                                                       // selected value visible in the form

// ---- collapse the editor (Submit stays in the bar) and show the whole graph: 6 specialists + escalate
await click('css:button[aria-label="Collapse input editor"]');
await sleep(300);
await fit();
rec.mark('graph_fit');
await sleep(1000);

// ---- Submit: run executes. Studio re-centres the graph when the run starts, so frame the part of the route that will run afterwards.
await click('css:button[aria-label="Submit"]');
rec.mark('submit');
await sleep(700);
rec.mark('thread', { threadId: new URL(location.href).searchParams.get('threadId') });
await frameRoute();                                                      // zoom so triage .. escalate fill the pane
rec.mark('graph_framed');
vp('framed');
await waitFor(() => submitBtn(), 60000, 'run finished (Submit is back)');
rec.mark('run_finished');
vp('finished');
await sleep(900);

// ---- finished run: walk the steps so the graph lights each node
rec.mark('finished_graph');
for (const name of ['load_ticket', 'triage', 'escalate']) {
  await studioHover(studioStep(name));
  rec.mark('step_' + name);
  await sleep(1300);
}
studioUnhover();
await hover(byText('TURN 1') || document.body);
await sleep(1200);
rec.mark('end');
