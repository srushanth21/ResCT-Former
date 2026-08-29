Here is a step-by-step guide to building a clean, publication-ready architecture diagram for **ResCT-Former** using [app.diagrams.net (Draw.io)](https://app.diagrams.net/).

---

## 1. Canvas Setup & General Styling

Before placing shapes, configure Draw.io for an academic paper:

1. **Page Settings (Right Panel):**
* Uncheck **Page View** (gives an infinite canvas).
* Check **Grid** and set grid size to `10 pt` (helps with snapping and perfect alignment).


2. **Font Selection:**
* Select all text components using **Times New Roman** or **Helvetica/Arial** (10–12 pt for block labels, 8–9 pt for tensor dimensions). This matches the IEEE document font style.


3. **Color Palette (Muted / Academic Scheme):**
* **Backbone / CNN Blocks:** Soft Blue (`#DAE8FC`, border `#6C8EBF`)
* **CTA (Cross-Temporal Attention):** Warm Amber / Orange (`#FFE6CC`, border `#D79B00`)
* **CEFF (Fusion):** Mint Green (`#D5E8D4`, border `#82B366`)
* **Progressive Decoder:** Lavender / Purple (`#E1D5E7`, border `#9673A6`)
* **Change Mask:** Dark Slate / Black & White (`#F5F5F5` with high-contrast border)



---

## 2. Layout Structure (Left to Right)

Organize the canvas into **5 horizontal stages** across the page:

```
[Inputs] ──► [Siamese ResNet-34] ──► [CTA Modules] ──► [CEFF Fusion] ──► [Decoder] ──► [Mask]

```

---

### Step 1: Input Images & Siamese Backbone

1. **Inputs ($T_1$ and $T_2$):**
* Place two square rectangles on the far left, stacked vertically (top for $T_1$ Pre, bottom for $T_2$ Post).
* *Labeling:* `Image T1 (Pre)` and `Image T2 (Post)`, with subtext `$3 \times 256 \times 256$`.


2. **Siamese ResNet-34 Streams:**
* For the top stream ($T_1$), create **4 horizontal rectangles** in a row:
* `C1 (64ch, 64×64)`
* `C2 (128ch, 32×32)`
* `C3 (256ch, 16×16)`
* `C4 (512ch, 8×8)`


* Duplicate this exact row below for $T_2$.
* *(Optional Visual Cue)* Make each subsequent stage slightly taller/thicker or darker to represent increasing channel depth ($64 \to 512$).


3. **Shared Weight Container:**
* Draw a large dashed rectangle around both rows.
* Label the top of this container: **`Shared ResNet-34 Backbone (Pretrained)`**.



---

### Step 2: Bidirectional Cross-Temporal Attention (CTA)

1. **CTA Blocks:**
* Place an orange rounded rectangle between the top `C3` and bottom `C3` blocks. Label it **`CTA (Stage 3)`**.
* Place a second orange rounded rectangle between the top `C4` and bottom `C4` blocks. Label it **`CTA (Stage 4)`**.


2. **Cross-Connections:**
* Draw arrows from top `C3` and bottom `C3` entering the `CTA (Stage 3)` block.
* Draw output arrows from `CTA (Stage 3)` producing the refined $\hat{F}_{pre}^3$ and $\hat{F}_{post}^3$ features.
* Repeat the exact same connection for `C4`.



---

### Step 3: CEFF Fusion (All 4 Stages)

1. Place **4 vertical boxes** to the right of the backbone/CTA stage, aligned with stages 1 through 4:
* `CEFF Stage 1` (receives raw $C_1^{pre}$ and $C_1^{post}$) $\to$ outputs $f_1 (64\text{ch})$
* `CEFF Stage 2` (receives raw $C_2^{pre}$ and $C_2^{post}$) $\to$ outputs $f_2 (128\text{ch})$
* `CEFF Stage 3` (receives attended $\hat{F}_{pre}^3$ and $\hat{F}_{post}^3$) $\to$ outputs $f_3 (256\text{ch})$
* `CEFF Stage 4` (receives attended $\hat{F}_{pre}^4$ and $\hat{F}_{post}^4$) $\to$ outputs $f_4 (512\text{ch})$


2. Add a small formula note above or inside the CEFF boxes:

$$\text{CEFF} = \text{Dropout}(\text{ReLU}(\text{BN}(\text{Conv}_{3\times3}(\vert{}F_{pre} - F_{post}\vert{}))))$$



---

### Step 4: Progressive U-Net Decoder & Deep Supervision

1. **Hierarchical Decoder Tower:**
* Draw a vertical set of connected blocks (purple) representing the $2\times$ upsampling stages.
* Show projection layers bringing each $f_i$ to $256\text{-dim}$ (`embed_dim=256`).
* Connect arrows from $f_4 \to f_3 \to f_2 \to f_1$ with addition nodes ($\oplus$) and $2\times$ upsample blocks.


2. **Deep Supervision (Auxiliary Heads):**
* Draw dashed arrows branching out from $f_2$ and $f_3$ leading to small boxes labeled **`Aux Loss (0.2)`** and **`Aux Loss (0.4)`**.


3. **Final Classification Head:**
* Connect the output of the decoder to a block: `Conv(256→128) → BN → ReLU → Conv(128→1) → 4× Upsample`.



---

### Step 5: Output Change Map

* Place a final square on the far right.
* Fill with a high-contrast binary building change mask (black background with crisp white building contours).
* Label: **`Binary Change Map M`** ($1 \times 256 \times 256$).

---

## 3. Draw.io Pro-Tips for Clean Alignments

* **Distribute Evenly:** Select multiple blocks $\to$ click **Arrange** (right panel) $\to$ **Distribute** $\to$ **Horizontal/Vertical**.
* **Clean Connectors:** Set arrow styles to **Orthogonal** (elbow arrows) with `Rounded` corners to keep intersecting lines legible.
* **Add a Legend:** Add a small box in the bottom corner explaining:
* Solid line = Feature flow
* Dashed line = Deep supervision / Auxiliary flow
* $\oplus$ = Element-wise addition



---

## 4. Exporting for LaTeX / Overleaf

1. Go to **File $\to$ Export as $\to$ PDF...**
2. In the export dialog:
* Check **Crop** (trims whitespace tightly around your diagram).
* Ensure **Transparent Background** is unchecked (standard white background).


3. Save the file as `resct_former_architecture.pdf`.
4. Upload `resct_former_architecture.pdf` directly into your LaTeX project directory and include it in your document:

```latex
\begin{figure*}[htbp]
    \centering
    \includegraphics[width=\textwidth]{resct_former_architecture.pdf}
    \caption{Overall architecture of the proposed ResCT-Former framework...}
    \label{fig:architecture}
\end{figure*}

```