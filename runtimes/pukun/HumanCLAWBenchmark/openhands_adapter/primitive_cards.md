HumanCLAW native whole-body primitives. Import primitive_api as p.

- p.get_task_context(): original instruction, elapsed steps and native 100-step cap.
- p.observe(): expose the current ego RGB image; the harness attaches it to the NEXT model request.
- p.walk_forward(speed="slow", visible_state=""): speed slow/normal/fast maps to native 0.2/0.4/0.6 m conditioning.
- p.turn(direction="left", degrees=30, visible_state=""): left/right, 10–120 degrees; native motion may be obstructed.
- p.step_back(distance=0.25, visible_state=""): 0.10–0.60 m.
- p.side_step(direction="left", distance=0.25, visible_state=""): left/right, 0.10–0.50 m.
- p.climb_up(visible_state=""): native [0.28, 0.30] conditioning.
- p.climb_down(visible_state=""): native [0.20, 0.40] conditioning.
- p.sit(height=0.5, visible_state=""): estimated seat height 0.15–0.85 m.
- p.stop(visible_state=""): commit the current pose and irreversibly end the episode; no extra stand motion.

When you believe the instruction is fulfilled, call p.stop() in a final cell. Native navigation and interaction success require this explicit Stop; exhausting a budget does not commit the pose. For sitting tasks, sit, inspect the resulting image, and then Stop when satisfied. Reserve a step and a cell for Stop.

First cell: observe and yield. Thereafter execute at most ONE movement/stop per cell and yield to see its result. Variables persist between cells. Every motion is one native 15-frame chunk, not a teleport or path planner. Judge achieved movement from the next image; collision can prevent commanded movement. Each action's visible_state should describe only what you actually see in the latest attached frame, explicitly acknowledging the goal object when visible. Do not guess its presence. Native FindSR pairs this acknowledgement with that same image. Native scoring is private and runs after the episode. No reset, target coordinates, semantic labels, teleport, ground-truth geometry, or success oracle is exposed.
