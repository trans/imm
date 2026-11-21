# The parallel path against the per-segment loop with a window wider than an
# update (max_memory 8 and 12 against 5 segments an update, 3 updates: the
# window reaches into carried records and, at 12, rolls over in the last
# update); one and two blocks; dropout off in train mode.
import sys, os
sys.path[0:0] = [os.getcwd(), os.path.join(os.getcwd(), "tests"), os.path.join(os.getcwd(), "scripts")]
import torch
import test_train_streams as T
runner = T.runner
ids, rows = T.corpus()
segs = runner.split_streams(T.ragged_segments(len(ids) - 1), 3)
for max_memory in (8, 12):
    for layers in (1, 2):
        model = T.make_model(layers=layers, dropout=0.0, **T.TrainStreamsTest.PARALLEL)
        model.train(True)
        got = []
        for parallel in (True, False):
            model.parallel_memory = lambda carry_hidden, parallel=parallel: parallel
            model.zero_grad()
            memory, hidden, losses = None, None, []
            for u in range(3):
                part = [s[u * 5 : (u + 1) * 5] for s in segs]
                spans = [(p[0][0], p[-1][1]) for p in part]
                loss, _, memory, hidden, _ = model.segment_loss_streams(
                    ids, part, memory, max_memory, carried_hidden=hidden, carry_hidden=True, feedback_state="gru",
                    **{k: runner.prior_rows_for_spans(v, spans, torch.device("cpu")) for k, v in rows.items()},
                )
                loss.backward()
                losses.append(loss.detach())
                memory, hidden = (memory[0].detach(), memory[1]), hidden.detach()
            got.append((torch.stack(losses), T.grads(model), memory))
        (la, ga, ma), (lb, gb, mb) = got
        torch.testing.assert_close(la, lb, rtol=1e-5, atol=1e-5)
        for name in ga:
            torch.testing.assert_close(ga[name], gb[name], rtol=1e-4, atol=1e-5, msg=name)
        torch.testing.assert_close(ma[0], mb[0], rtol=1e-5, atol=1e-5)
        assert torch.equal(ma[1], mb[1]) and ma[0].shape[1] == min(max_memory, 15), ma[0].shape
        print(f"max_memory={max_memory} layers={layers}: parallel = loop, records held {ma[0].shape[1]}, losses {[round(float(x), 4) for x in la]}")
