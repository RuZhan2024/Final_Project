# Demo Script

This document provides a narration script for the software demonstration video.
It is designed for a short local demo that shows the system running and
demonstrates one meaningful end-to-end feature.

## Full Narration Script

### 1. Opening

“This project is Safe Guard, a pose-based fall detection system. It includes a
frontend monitoring interface and a backend inference service. In this demo, I
will show the system running locally and demonstrate one end-to-end feature.”

### 2. System running

“The frontend and backend are already running locally for this demonstration. I
am using the local setup to keep the run stable and reproducible.”

### 3. Move to Monitor

“I am now on the Monitor page, which is the main interface for running and
observing the system.”

### 4. Replay Mode

“I start with Replay Mode, which is the main demonstration path. Here I choose
a replay clip as the system input.”

### 5. Select clip

“This replay clip is the input to the detector. Using replay mode allows the
behaviour to be demonstrated in a controlled and repeatable way.”

### 6. Play replay

“Now I click Play Replay so the system can process the clip.”

### 7. Explain output

“The system is now processing the input and producing fall-related prediction
output. Here the current prediction is shown. The timeline updates as
prediction windows are processed. This panel shows the runtime model and
configuration used for the current run.”

### 8. End-to-end statement

“This demonstrates one meaningful end-to-end feature: the system takes an input
clip, performs inference, and returns visible results through the monitoring
interface.”

### 9. Realtime Mode

“I will also briefly show Realtime Mode. Here the system takes live camera
input through the browser.”

### 10. Live preview explanation

“The live preview is optional and is shown here only to make the realtime input
visible during the demonstration. The fall detection pipeline does not depend
on the preview being shown.”

### 11. Start realtime

“Now I start realtime monitoring. The interface continues to update as the
system processes live input.”

### 12. Optional privacy line

“In privacy-sensitive use, the preview can remain off unless it is explicitly
needed.”

### 13. Optional Event History

“The project also includes an event history workflow when database-backed mode
is enabled. This page is used to review stored events.”

### 14. Closing

“In summary, this software artefact demonstrates a working pose-based fall
detection system. The main feature shown here is end-to-end monitoring from
input to prediction output. The project includes replay monitoring, realtime
monitoring, and a runnable frontend-backend system.”

## Shorter Version

If a shorter narration is preferred:

“This project is a pose-based fall detection system with a frontend and backend
running locally. I start with Replay Mode, select a replay clip, and play it as
the input to the detector. The system produces prediction output here, the
timeline updates as windows are processed, and this panel shows the runtime
model information. This demonstrates one end-to-end feature. I then briefly
show Realtime Mode, where the system processes live camera input through the
browser. The live preview is optional and is shown only for demonstration. In
summary, this is a working software artefact that demonstrates the core fall
detection workflow.”

## Recording Tips

- pause briefly after each action
- keep the mouse still when referring to a panel
- speak in short sentences
- if a step is unstable, cut it rather than forcing it into the demo
