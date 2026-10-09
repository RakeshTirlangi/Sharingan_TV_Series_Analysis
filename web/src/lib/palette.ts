// Validated categorical palette (dark-surface steps), assigned in fixed order and
// never cycled: a 9th community folds into OTHER. Colour follows the entity.
export const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
export const OTHER = "#6f6e69";
export const seriesColor = (index: number) => (index >= 0 && index < SERIES.length ? SERIES[index] : OTHER);

// Diverging blue <-> red with a neutral grey midpoint (dark surface).
const BLUE = [57, 135, 229], RED = [230, 103, 103], MID = [56, 56, 53];
const mix = (a: number[], b: number[], t: number) => `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
/** value in [-1, 1] -> colour; 0 is neutral */
export const diverging = (v: number) => (v < 0 ? mix(MID, BLUE, Math.min(1, -v)) : mix(MID, RED, Math.min(1, v)));

// Sequential single hue (blue), dark surface: low -> recedes, high -> bright.
export const sequential = (t: number) => mix([26, 26, 25], [57, 135, 229], Math.max(0, Math.min(1, t)));

export const BRAND = "#e5383b";
