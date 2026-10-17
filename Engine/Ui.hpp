#pragma once

// Which tab of the main window is being built.
//
// Every subsystem draws into more than one of them, so ShowImgui takes a
// section rather than owning a window: the renderer's shadow sliders belong
// next to the sky's, and its cascade-cache counters belong next to the scene's
// draw counts, which are in two different places in the UI and two different
// files here.
//
// The split that matters is Performance against everything else. A number you
// read and a control you change want opposite things from a layout -- the
// numbers want to be visible all at once while a frame is being measured, and
// the controls want to be out of the way -- and mixing them meant every cost in
// the frame was behind a different collapsing header.

namespace NeuralModelIntegrateTestbed {

enum class UiSection {
    // What the frame looks like and how it is drawn: the view selector,
    // shadows, the sky, the display transform.
    Render,
    // The asset: the camera, its hierarchy, its animations, its images.
    Scene,
    // The neural models and the motion pipeline that drives them.
    Neural,
    // Every cost and timing in the process, from anywhere in it.
    Performance,
};

}  // namespace NeuralModelIntegrateTestbed
