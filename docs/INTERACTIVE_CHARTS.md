# Interactive result charts

The project page renders the five domain figures as native SVG and HTML charts. Their values follow the retained paper results; model colours come from `model-style.js`, and the heatmaps reuse the current editable Spatial figure’s 100-colour warm ramp.

- Select a Spatial column heading to read the task definition and compare all ten models as bars with run SD. Hover, tap, or focus any data mark for its value and scope.
- Camera-condition and source buttons switch measured series without changing axis scales.
- Bars grow from zero when they enter the viewport. Heatmap cells and scatter points reveal on entry. Leaving the viewport fully allows replay from either scroll direction. Reduced-motion preferences, keyboard focus, printing, and hidden tabs stop motion.
- Narrow screens keep legible labels and use local horizontal scrolling. Model names remain fixed beside heatmaps.

`public/data/report/domain-charts.json` records the chart inputs and source hashes. `scripts/build_domain_chart_data.py` regenerates it from the retained manuscript evidence workspace; `--check` compares the generated result without writing. Spatial means, run SD, and case counts are copied without rounding. Tracing and polygon values retain the precision used in the released paper tables. Planning partitions all 157 tasks exactly once; CALVIN retains all 100 adopted episodes.

Axis definitions were checked against the original question snapshot. **Actor** uses the actor’s own orientation; **Facing** asks which side of an object faces another object; **Hard** asks for a four-way quadrant from a specified position and facing direction in a video scene.

Validation: `python3 -m unittest discover -s tests -p 'test_public_domain_charts.py'` and `node tests/check_figure_motion.cjs`. The existing case, execution, comparison, and media checks remain in place.
