"""Experiment registry. Each entry: id, group, cfg (ffic command-line), mechanism, FunnelBrain source, hypothesis.
FunnelBrain sources refer to raajvamsy/funnel-brain: docs/FUNNEL_BRAIN_HYPOTHESIS.md (sections) and funnel/*.py."""

REFERENCE = "ref_best"
B = "ltpcrewb"  # the current best model (nested funnels + RLS + colour taps + binary CM coder)

EXPERIMENTS = [
    dict(id="ref_best", group="reference", cfg=B,
         mechanism="Current best model before this round: nested funnels root->activity->spin->level, RLS predictors with colour taps, parallel-ball blend, error-energy mouth, binary context-mixing residual coder.",
         funnelbrain="Funnels, nesting with inheritance (fractal meta-funnels), parallel balls + global workspace, PE learning, fatigue.",
         hypothesis="Reproduces the previous result (10,600,865 B, +1.1% vs JXL e9) byte-for-byte after the rewrite."),

    # ---- the continuous funnel field ------------------------------------------------------------------------------
    dict(id="field", group="field", cfg=B + " field=1",
         mechanism="Replace the fixed spin/level grid by a continuous FUNNEL FIELD: 7-D descriptor (activity, error energy, wide activity, orientation x strength (2-D), brightness, cross-plane surprise). A pixel lands in the nearest funnel; if outside every mouth a new funnel is created there, inheriting from the nearest. Prediction blends the 3 most resonant funnels.",
         funnelbrain="funnel/field.py find_or_create_funnel (MERGE_RADIUS), funnel/physics.py resonance() (radial x depth pull x fatigue), embed_bytes multi-scale transduction.",
         hypothesis="Learned classes fit the image better than a fixed quantisation grid; expect -1 to -3%."),
    dict(id="field_m08", group="field", cfg=B + " field=1 merge=0.08",
         mechanism="Funnel field with a smaller merge radius (more, narrower funnels).", funnelbrain="MERGE_RADIUS (field.py).",
         hypothesis="Finer concepts help on large images if inheritance keeps learning cost low."),
    dict(id="field_m18", group="field", cfg=B + " field=1 merge=0.18",
         mechanism="Funnel field with a larger merge radius (fewer, wider funnels).", funnelbrain="MERGE_RADIUS (field.py).",
         hypothesis="Fewer funnels learn faster; may win on small/noisy images."),
    dict(id="field_k1", group="field", cfg=B + " field=1 kres=1",
         mechanism="Funnel field, only the landing funnel predicts (no resonance blend of neighbours).", funnelbrain="Single landing vs resonance with several mouths.",
         hypothesis="Resonance blending of 3 funnels beats a single landing funnel."),
    dict(id="field_sleep", group="field", cfg=B + " field=1 sleep=16",
         mechanism="Funnel field + SLEEP every 16 rows: FISSION of busy funnels whose landings spread wider than their mouth (split along the widest descriptor axis, both children inherit) and PRUNE of idle young funnels.",
         funnelbrain="Sleep MDL, fission_mixed(), sleep_mdl() in field.py; hypothesis doc section 5.",
         hypothesis="Splitting where the data shows two modes sharpens predictors; expect a small gain over field."),
    dict(id="field_f96", group="field", cfg=B + " field=1 fmax=96",
         mechanism="Funnel field capped at 96 funnels per plane.", funnelbrain="Saturation of the field.",
         hypothesis="Shows how many concepts the picture actually needs; also faster."),

    # ---- association, memory --------------------------------------------------------------------------------------
    dict(id="bounce", group="association", cfg=B + " bounce=1",
         mechanism="BOUNCE PATH context: the funnel ids where the W and N pixels landed plus the current funnel id, hashed, become an extra residual-coder context.",
         funnelbrain="Associations emerge from bounce paths between funnels; no stored edges (hypothesis doc section 2, 4).",
         hypothesis="Funnel transitions carry texture continuity that single-pixel contexts miss; small gain."),
    dict(id="field_bounce", group="association", cfg=B + " field=1 bounce=1",
         mechanism="Bounce path over the continuous funnel field (learned funnel ids instead of grid cells).", funnelbrain="Bounce path + field.",
         hypothesis="Learned funnel ids make better path contexts than grid cells."),
    dict(id="hippo", group="memory", cfg=B + " hippo=1",
         mechanism="HIPPOCAMPUS: an episodic table keyed by the quantised shape of 7 neighbours (+ G surprise for R/B) stores what followed that shape; recall is an extra ball, an RLS tap and a coder context.",
         funnelbrain="Hippocampus lens stores ball-path episodes; cerebellum predicts the landing from recalled episodes (organs/cerebellum.py).",
         hypothesis="Repeated local shapes (texture, JPEG blocks) get predicted from memory; helps Pexels more than Kodak."),

    # ---- plasticity, salience, workspace -----------------------------------------------------------------------------
    dict(id="lampe", group="plasticity", cfg=B + " lampe=1",
         mechanism="Online PE reshapes geometry: each funnel's RLS forgetting factor follows its surprise (big error => shorter memory, calm => longer).",
         funnelbrain="reshape_on_surprise() in field.py: surprise widens/deepens the funnel immediately.",
         hypothesis="Adapts faster at region boundaries and remembers longer inside regions; small gain."),
    dict(id="sal_robust", group="plasticity", cfg=B + " sal=-1",
         mechanism="Salience damped: surprising pixels get LESS learning weight (robust regression).", funnelbrain="Amygdala amplifies energy on salience (inverted here as a control).",
         hypothesis="Outliers (edges, noise spikes) corrupt LS predictors; damping them helps."),
    dict(id="sal_amp", group="plasticity", cfg=B + " sal=1",
         mechanism="Salience amplified: surprising pixels get MORE learning weight.", funnelbrain="Amygdala lens: amplify ball energy on salience.",
         hypothesis="Faster reaction to new structure; probably hurts on noise."),
    dict(id="ws1", group="workspace", cfg=B + " ws=1", mechanism="Global workspace temperature 1 (softer blend).", funnelbrain="Global workspace picks the winning path.",
         hypothesis="Softer blend averages out predictor noise."),
    dict(id="ws4", group="workspace", cfg=B + " ws=4", mechanism="Global workspace temperature 4 (sharper blend).", funnelbrain="Global workspace picks the winning path.",
         hypothesis="Closer to a single winner; better when one funnel is clearly right."),
    dict(id="ws0", group="workspace", cfg=B + " ws=0", mechanism="Winner takes all: only the most skilled funnel on the path predicts.", funnelbrain="One winning ball path = conscious thought.",
         hypothesis="Literal FunnelBrain workspace; expected to lose to blending."),
    dict(id="replay", group="sleep", cfg=B + " replay=1",
         mechanism="SLEEP REPLAY: after each row, the landing funnels re-learn that row once more (the decoder can replay identically).",
         funnelbrain="Offline replay during sleep (hypothesis doc section 2, 5).",
         hypothesis="Extra epochs speed learning of young funnels; may overfit (RLS already optimal for one pass)."),
    dict(id="ms", group="transduction", cfg=B + " ms=1",
         mechanism="MULTI-SCALE transduction: wide-window activity (8 samples 3-6 px away) as an extra coder context with local activity and error energy.",
         funnelbrain="Multi-scale byte transduction windows 4 / 64 / 1024 (transduce.py).",
         hypothesis="Region-level texture tells the coder how noisy the neighbourhood is beyond 2 px; small gain."),

    # ---- combinations ------------------------------------------------------------------------------------------------
    dict(id="combo_a", group="combo", cfg=B + " field=1 sleep=16 bounce=1 hippo=1 ms=1",
         mechanism="Field + sleep + bounce + hippocampus + multi-scale together.", funnelbrain="Most of the brain at once.",
         hypothesis="Gains add up if the mechanisms capture different structure."),
]

R = B + " sal=-1"   # round-2 base: robust salience
EXPERIMENTS += [
    # ---- round 2: how strong should salience damping be? ----------------------------------------------------------
    dict(id="sal_p2", group="salience", cfg=R + " salp=2", mechanism="Robust salience, weight 1/surprise^2 (clamped at 4).", funnelbrain="Amygdala salience (inverted: surprise is distrusted).",
         hypothesis="Stronger damping of outliers helps further."),
    dict(id="sal_p05", group="salience", cfg=R + " salp=0.5", mechanism="Robust salience, weight 1/sqrt(surprise).", funnelbrain="Amygdala salience.",
         hypothesis="Milder damping; tells where the optimum is."),
    dict(id="sal_max8", group="salience", cfg=R + " salmax=8", mechanism="Robust salience, clamp at 8 instead of 4.", funnelbrain="Amygdala salience.",
         hypothesis="Very large surprises should be damped even more."),
    dict(id="sal_max2", group="salience", cfg=R + " salmax=2", mechanism="Robust salience, clamp at 2.", funnelbrain="Amygdala salience.",
         hypothesis="Only mild outliers matter."),
    dict(id="sal_bias", group="salience", cfg=R + " salbias=2", mechanism="Robust salience + bias update clipped at 2x the funnel's usual error.", funnelbrain="Amygdala salience on the bias too.",
         hypothesis="Outliers also corrupt the bias; clipping helps slightly."),
    # ---- round 2: new FunnelBrain mechanisms on the new base ------------------------------------------------------
    dict(id="r_register", group="registers", cfg=R + " register=1",
         mechanism="REGISTERS: an R or B funnel is born from G's trained funnel at the same address (spatial weights inherited, colour taps fresh) instead of from its own parent.",
         funnelbrain="MainRegister / SubRegister: nested meta-funnels (parent_meta_id), language register tree (hypothesis doc section 11).",
         hypothesis="Each plane starts from scratch today; inheriting G's spatial knowledge saves learning cost on R and B."),
    dict(id="r_grav05", group="gravity", cfg=R + " grav=0.5", mechanism="GRAVITY / depth pull: workspace weight x (1 + 0.5 log(2 + exposures)).",
         funnelbrain="physics.py GRAVITY and resonance depth_pull: deep funnels pull balls in.", hypothesis="Experienced funnels deserve extra trust beyond their running error."),
    dict(id="r_curve", group="spin", cfg=R + " curve=1", mechanism="SPIN AS CURVATURE: an extra ball continuing the local edge trend with its curvature (second difference) from N and W directions.",
         funnelbrain="Wave-ball spin: Magnus term curves the trajectory, never straight (physics.py integrate_step).",
         hypothesis="Curved edges and gradients are predicted better than by straight-line balls."),
    dict(id="r_field_m08", group="field", cfg=R + " field=1 merge=0.08", mechanism="Robust base + funnel field, small mouths.", funnelbrain="Field + salience.",
         hypothesis="The two best round-1 effects add."),
    dict(id="r_field_m08_k1_sleep", group="field", cfg=R + " field=1 merge=0.08 kres=1 sleep=16 fmax=128",
         mechanism="Robust base + field (small mouths, landing only, sleep fission/prune, cap 128).", funnelbrain="Field + sleep + salience.",
         hypothesis="Best field settings from round 1 combined."),
    dict(id="r_ms_bounce", group="combo", cfg=R + " ms=1 bounce=1", mechanism="Robust base + multi-scale + bounce contexts.", funnelbrain="Transduction + association.",
         hypothesis="Small neutral-to-positive effects add up."),
]

R2 = B + " sal=-1 salp=2"   # round-3 base
EXPERIMENTS += [
    dict(id="sal_p3", group="salience", cfg=B + " sal=-1 salp=3", mechanism="Robust salience, weight 1/surprise^3.", funnelbrain="Amygdala salience (inverted).", hypothesis="Optimum may be beyond p=2."),
    dict(id="sal_p2_max8", group="salience", cfg=R2 + " salmax=8", mechanism="Robust salience p=2, clamp 8.", funnelbrain="Amygdala salience.", hypothesis="Damping very large surprises harder helps."),
    dict(id="sal_p2_max16", group="salience", cfg=R2 + " salmax=16", mechanism="Robust salience p=2, clamp 16.", funnelbrain="Amygdala salience.", hypothesis="Where the clamp stops mattering."),
    dict(id="salall3", group="salience", cfg=R2 + " salall=3", mechanism="Salience everywhere: skill and mouth estimators clip outliers at 3x usual error.", funnelbrain="Amygdala salience applied to every adaptive estimate.",
         hypothesis="Outliers inflate the residual scale and distort skill weights; clipping helps."),
    dict(id="salall2", group="salience", cfg=R2 + " salall=2", mechanism="Salience everywhere, clip at 2x.", funnelbrain="Amygdala salience.", hypothesis="Stronger clipping."),
    dict(id="stack1", group="combo", cfg=R2 + " salmax=8 salbias=2 register=1 grav=0.5",
         mechanism="Stack: robust salience p2 (clamp 8, bias clip) + registers + gravity.", funnelbrain="Amygdala + registers + gravity.", hypothesis="Independent small gains add (~-0.3%)."),
    dict(id="stack2", group="combo", cfg=R2 + " salmax=8 salbias=2 register=1 grav=0.5 field=1 merge=0.08 kres=1 sleep=16 fmax=128",
         mechanism="stack1 + funnel field (small mouths, landing only, sleep, cap 128).", funnelbrain="+ funnel field and sleep.", hypothesis="Field adds ~-0.3% on top."),
    dict(id="stack3", group="combo", cfg=R2 + " salmax=8 salbias=2 register=1 grav=0.5 field=1 merge=0.08 kres=1 sleep=16 fmax=128 ms=1 bounce=1",
         mechanism="stack2 + multi-scale + bounce contexts.", funnelbrain="+ transduction scales + bounce association.", hypothesis="Small extra gain."),
]

S3 = R2 + " salmax=8 salbias=2 register=1 grav=0.5 field=1 merge=0.08 kres=1 sleep=16 fmax=128 ms=1 bounce=1"   # = stack3
EXPERIMENTS += [
    dict(id="stack4", group="combo", cfg=S3.replace("salp=2", "salp=3") + " salall=2", mechanism="stack3 with salience p=3 and salience everywhere (clip 2x).",
         funnelbrain="Amygdala salience, fully.", hypothesis="The two best single salience settings add ~-0.3% to stack3."),
    dict(id="acc", group="coder", cfg=S3 + " apm=1", mechanism="stack3 + ACC: secondary probability correction (APM) by error energy x ball disagreement.",
         funnelbrain="ACC lens computes PE on bounce surprise (hypothesis doc section 6).", hypothesis="Standard refine stage; expect -0.5 to -1.5%."),
    dict(id="mix2", group="coder", cfg=S3 + " mix2=1", mechanism="stack3 + second coder mixer selected by surprise level.",
         funnelbrain="A second global workspace view.", hypothesis="Two views of the same evidence average out; expect -0.3 to -1%."),
    dict(id="stack5", group="combo", cfg=S3.replace("salp=2", "salp=3") + " salall=2 apm=1 mix2=1", mechanism="stack4 + ACC + second mixer.",
         funnelbrain="Everything that helped so far.", hypothesis="Gains add."),
]

S5 = S3.replace("salp=2", "salp=3") + " salall=2 apm=1 mix2=1"   # = stack5, current best
EXPERIMENTS += [
    dict(id="mline", group="colour physics", cfg=S5 + " mline=1",
         mechanism="Material lines: R and B get an extra ball (and RLS tap) from a line R ~ a*G + b fitted on 6 decoded neighbours (ridge 20 toward slope 1).",
         funnelbrain="Colour formation: pixel = light x material x shading; a material under one light moves along a line in colour space as shading changes (dichromatic model).",
         hypothesis="Offline test: local material line cuts R prediction error 12% on Kodak (1.82 -> 1.61) but +9% on Pexels; expect a gain on Kodak, neutral on Pexels."),
    dict(id="mline2", group="colour physics", cfg=S5 + " mline=2", mechanism="Material lines, B also fitted against R (two line balls for B).",
         funnelbrain="Colour formation (both earlier planes).", hypothesis="R carries extra material information for B."),
    dict(id="mline_w12", group="colour physics", cfg=S5 + " mline=1 mwin=12", mechanism="Material lines fitted on 12 neighbours.", funnelbrain="Colour formation.",
         hypothesis="A larger window gives a steadier line, worse at material edges."),
    dict(id="mline_l5", group="colour physics", cfg=S5 + " mline=1 mlam=5", mechanism="Material lines, weaker ridge (slope free-er).", funnelbrain="Colour formation.",
         hypothesis="Free slope fits shading better but is noisier."),
    dict(id="mline_l80", group="colour physics", cfg=S5 + " mline=1 mlam=80", mechanism="Material lines, strong ridge (slope near 1 unless the evidence is strong).", funnelbrain="Colour formation.",
         hypothesis="Closer to the difference model; tells whether the slope matters."),
]
