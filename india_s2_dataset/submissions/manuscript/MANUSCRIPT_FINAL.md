---
title: "Remote Sensing Image Change Caption Generation for Indian Sentinel-2 Imagery: Model Transfer, Dataset Construction and Automatic Labelling"
---

**Sankalp Chaturvedi (2025201083), KSPSVLN Siddardha Kumar Kavuri (2025201061), Rinkesh Verma (2025201070)**

**Mentor:** Rama Chandra Prasad · Independent Study, Monsoon 2026

**Code:** [GitHub repository](https://github.com/Sankalp0109/Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection) · **Dataset:** [India Sentinel-2 Change Pairs](https://www.kaggle.com/datasets/kspsvlnsiddardha/india-sentinel2-change-pairs) · **Checkpoints:** [Hugging Face](https://huggingface.co/Kspsvln/IS) · **All links:** [Appendix B](#appB)

**Contents:** [Abstract](#sec-abstract) · [1. Introduction](#sec-intro) · [2. Study area and data](#sec-data) · [3. Methodology](#sec-method) · [4. Results and discussion](#sec-results) · [5. Limitations and future scope](#sec-limits) · [6. Conclusions](#sec-concl) · [Acknowledgements](#sec-ack) · [References](#sec-refs) · [Appendix A: Supplementary tables and figures](#appA) · [Appendix B: Data, code and model links](#appB)

# Abstract {#sec-abstract}

Change captioning asks a model to look at two satellite images of the same place, taken at different times, and write a sentence about what changed. Public datasets for this task use sub-metre imagery of a few foreign cities, and none covers India or the 10 m Sentinel-2 imagery that is freely available here. We approached the problem in three parts, each started because the previous one did not succeed. First, we trained and compared seven captioning designs on LEVIR-CC in one shared pipeline. Replacing a CNN trained from scratch with a frozen RemoteCLIP encoder gave the largest gain (BLEU-4 from 0.303 to 0.328), while more elaborate fusion added at most 0.009. Fine-tuning on SECOND-CC improved that dataset only modestly and cut LEVIR-CC BLEU-4 to 0.12–0.15, a clear case of catastrophic forgetting; language-model decoders could not be trained to a usable state on the available GPUs. Second, two off-the-shelf captioners run on Indian Sentinel-2 pairs either denied change almost everywhere or produced empty or garbled text. Third, we built an India-wide dataset of 15,610 quality-screened Sentinel-2 pairs and tested whether large vision-language models could label land-use change in it. Across prompt designs, region hints, a rule-based change map, model sizes and a 1,000-pair validation against human labels, the best decision rule was correct on at most about half of its change calls, against the 90% needed to build a dataset automatically. The central difficulty is separating lasting land-use change from seasonal crop and water differences at 10 m. Scaling either route would produce a poor dataset; the screened dataset, 1,002 labelled pairs and review tools are released for human-annotated work.

**Keywords:** change captioning; Sentinel-2; vision-language models; India; dataset construction; land-use change

# 1. Introduction {#sec-intro}

Detecting what has changed on the ground between two dates is one of the oldest uses of satellite imagery. Early work compared spectral values pixel by pixel or classified each date separately and compared the maps, and a widely cited review had already grouped a large number of such techniques by two decades ago [@lu2004]. Deep learning moved the field toward learned change masks, and benchmarks such as LEVIR-CD, built from 0.5 m Google Earth imagery of buildings [@levircd], and OSCD, built from Sentinel-2 imagery of urban areas [@oscd], made it possible to train and compare such models. A change mask, however, only tells the user where the pixels differ. It does not say whether the difference is a new housing estate, a filled reservoir or a harvested field, and it gives nothing to a reader who is not a remote-sensing specialist.

Remote sensing image change captioning addresses this gap by producing a short natural-language description of the change instead of, or alongside, a mask [@hoxha][@levircc]. A caption such as "a road appears at the bottom and several houses replace the trees" can be indexed, searched and read by planners, journalists or local officials without image-analysis training. The main public benchmark for the task, LEVIR-CC, contains 10,077 image pairs of 256×256 pixels at 0.5 m resolution from Texas, each with five human-written captions [@levircc]. SECOND-CC extends the idea to a wider range of semantic land-cover changes [@secondcc]. Models trained on these datasets combine an image encoder that looks at both dates with a text decoder that writes the sentence. The model that accompanied LEVIR-CC, for example, extracts features from both dates with a pretrained convolutional network, relates them with dual-branch Transformer layers and passes the result to a Transformer decoder [@levircc]. Most later work keeps this encoder–decoder structure and varies how the two dates are compared, for instance through attention between them, differencing at several scales or contrastive objectives that tie image features to caption embeddings. Progress is usually reported as a few points of BLEU or CIDEr on the same benchmark, which raises a practical question that the benchmark itself cannot answer: how much of that progress survives a change of region, sensor and resolution?

These benchmarks were not designed with India in mind, and the mismatch is large. They use very high-resolution commercial imagery, while the imagery that is free and regularly available for the whole of India is Sentinel-2 at 10 m [@sentinel2]. They are dominated by planned suburban construction, while Indian landscapes are dominated by smallholder agriculture, dense and irregular settlements with varied roofing, mining, reservoirs and forest edges. Most importantly, Indian imagery changes strongly with the seasons. Fields that are green in one image can be bare or ploughed in the other, river and reservoir levels move with the monsoon, and winter haze over the plains can wash out an entire scene. A model that has only seen Texas suburbs has no reason to treat these differences as anything other than change. To our knowledge there is no change-captioning dataset for India at all. Resolution makes this harder than it first appears. At 10 m a single house occupies one or two pixels, a village road is narrower than a pixel and a new building plot is a handful of pixels in a 256×256 patch. Changes that are obvious at 0.5 m, such as a new roof or a widened lane, are close to invisible, while changes that are large at 10 m, such as a field changing colour, are often seasonal and of no interest. Whatever is learned on sub-metre imagery therefore has to be relearned, and the cues that separate interesting from uninteresting change are different.

There are two obvious ways to close this gap. The first is to take models trained on existing datasets and adapt them, either by fine-tuning or by using general vision-language models that have already seen a great deal of imagery. The second is to build an Indian dataset. Building one by hand is expensive: thousands of pairs have to be inspected, the change has to be judged and a sentence written for each. Recent large vision-language models (VLMs) can describe images and compare two of them [@qwen25vl][@qwen3vl], and remote-sensing assistants such as TEOChat have been trained specifically on temporal Earth-observation data [@teochat]. If such models could label image pairs reliably, a large Indian dataset could be produced at low cost and then used to train a small, fast captioner. Whether they can is an empirical question, and it depends on three abilities: detecting lasting land-use change, such as construction, new roads, mining, new water bodies or cleared vegetation; discriminating it from seasonal differences in crops, greenness and water level and from atmospheric effects such as haze; and describing what changed and where. The second of these turns out to be the hardest.

This study follows both routes and measures where each one fails. It has four objectives:

- **O1.** Build and compare change-captioning architectures on LEVIR-CC in a single controlled pipeline, and test whether they transfer to a second dataset.
- **O2.** Test whether existing captioners can describe change in Indian Sentinel-2 imagery without further training.
- **O3.** Construct a quality-screened, season-matched, India-wide Sentinel-2 bi-temporal dataset suitable for change captioning.
- **O4.** Measure, against human labels, whether vision-language models can label land-use change in this dataset accurately enough to build it automatically.

The work was carried out in that order between July and October 2026, and each part was started because the previous one did not succeed ([Table 3](#tab3)). The answer to O1 shaped the decision to look at off-the-shelf models (O2), and their failure led to the dataset (O3) and the labelling experiments (O4). We report negative results in detail, because they show which shortcuts do not work for this region and resolution and why. The main contributions of the work are: a controlled comparison of seven captioning designs under one training and evaluation protocol; evidence that transfer by sequential fine-tuning fails in a specific and measurable way; a reproducible pipeline and a released dataset of 15,610 season-matched Sentinel-2 pairs covering every state and union territory of India, with 1,002 labelled pairs; and a quantitative test, against blind human labels, of whether current vision-language models can label land-use change in such imagery.

The rest of the paper is organised as follows. [Section 2](#sec-data) describes the study area and the data. [Section 3](#sec-method) describes the captioning models, the dataset construction pipeline, the labelling experiments and the evaluation measures. [Section 4](#sec-results) presents and discusses the results of each part, together with the reason for each next step. [Section 5](#sec-limits) covers limitations and future work, and [Section 6](#sec-concl) concludes. Supplementary tables and figures are in [Appendix A](#appA), and every dataset, checkpoint and code link is collected in [Appendix B](#appB).

# 2. Study area and data {#sec-data}

## 2.1 Study area {#sec-area}

The study area is the whole of India: all 28 states and 8 union territories, about 3.29 million km². The country spans very different landscapes within one imaging system. The Indo-Gangetic plains are intensively farmed and suffer from heavy winter haze. The Deccan plateau mixes rain-fed agriculture, scrub and fast-growing towns. The western arid zone has desert and irrigated pockets. The Himalaya brings snow, glaciers and steep terrain, the north-eastern states are often under cloud, and the coasts and islands have large areas of open water. A dataset that covers all of these has to cope with very different failure modes in the imagery itself.

We sampled sites evenly rather than at places where change was expected. Our aim was a dataset representative of the country, not one tuned to construction hotspots. Using state boundaries from geoBoundaries [@geoboundaries], 1,200 sites were placed so that every state and union territory received at least twelve, or fewer where the territory is too small to hold twelve separate sites, with the remainder shared in proportion to area. Within each state, farthest-point sampling spread the sites as evenly as possible and kept them away from sites already placed across the border. Further sites were added later to replace those that yielded no usable imagery. In total, 1,669 sites produced at least one accepted image pair ([Figure 1](#fig1)). Most of the sites without accepted pairs lie in the high Himalaya, in persistently cloudy parts of the north-east and on islands, where snow, cloud or open water ruled out a usable pair.

![[]{#fig1}**Figure 1.** Study area: India with the sampled sites. Blue: sites with at least one accepted image pair (1,669); grey: sites without accepted pairs (221), mostly snow-covered, cloudy or open-water sites. State boundaries: geoBoundaries ADM1.](figures/fig1_study_area.png){width=85%}

## 2.2 Data {#sec-datasets}

### 2.2.1 Benchmark datasets

Two public change-captioning datasets were used for the model experiments. LEVIR-CC [@levircc] contains 10,077 pairs of 256×256-pixel images at 0.5 m resolution, split into 6,815 training, 1,333 validation and 1,929 test pairs. Each pair has five reference captions. Roughly half of the pairs show no change and carry captions such as "there is no difference", and most of the changed pairs involve new buildings and roads. We trained on the first caption of each pair and evaluated on the full test set. SECOND-CC [@secondcc] covers a wider range of semantic land-cover changes and was used as a second domain; its test set has 1,227 pairs. For models trained only on LEVIR-CC, the vocabulary contained 733 words that appear at least twice in the training captions; for models later fine-tuned on SECOND-CC, a shared vocabulary of 1,300 words was built from both datasets. Images were resized to 256×256 pixels for the CNN baseline, normalised with ImageNet statistics, and to 224×224 pixels with CLIP statistics for all models with a RemoteCLIP encoder.

### 2.2.2 Sentinel-2 imagery

All Indian imagery came from the Sentinel-2 Multispectral Instrument [@sentinel2]. We used Level-2A surface-reflectance products, which are generated by the Sen2Cor processor and include a scene-classification layer that flags cloud, cloud shadow, snow, water and missing data for every pixel [@sen2cor]. Scenes were taken from the reprocessed Collection-1 archive served through the Earth Search STAC API [@earthsearch]. Collection-1 applies a single processing baseline to all years, so a change in processing software cannot show up as a change on the ground. We used the true-colour (red, green, blue) product at 10 m resolution.

Each site has a before image from the 2019–20 season and an after image from the 2025–26 season. Seasons run from July to June, because the reprocessed archive is sparse before May 2019. To keep seasonal differences small, both images of a site come from the same region-specific dry-season window, and the two acquisition dates are at most 30 days apart in their position within that window. The windows and the cloud limits for each region are listed in [Table 1](#tab1). Most of the country uses November to March, the northern mountains use October to December to avoid heavy snow, and Ladakh uses the snow-free late summer.

For the pilot collection used in Part B, imagery of 60 hand-picked areas in five change categories (urban growth, deforestation, agriculture, coastal and infrastructure) was tiled into 512×512-pixel patches. A tile was kept only if its change magnitude exceeded a threshold, leaving 1,072 of 2,126 tiles; the collection took 1 h 53 min to build with 12 download workers ([Appendix Figure A5](#figA5)).

Table: []{#tab1}**Table 1.** Season windows and block-level cloud limits by region (July–June cycles: "2019" = July 2019–June 2020).

| Region | Months | Cloud limit | Reason |
|---|---|---|---|
| Most of India | Nov–Mar | 1% | Post-monsoon dry season |
| Indo-Gangetic plains (re-download) | Feb–Apr | 1% | After the winter smog |
| Tamil Nadu, Puducherry | Jan–Apr | 1% | North-east monsoon Oct–Dec |
| Kerala | Dec–Mar | 3% | Persistent cloud |
| North-east states | Nov–Mar | 3% | Persistent cloud |
| Jammu & Kashmir, Himachal, Uttarakhand | Oct–Dec | 1% | Clear sky before heavy snow |
| Sikkim, Arunachal Pradesh | Oct–Dec | 3% | Before snow; cloudy |
| Ladakh | Jul–Oct | 1% | Snow-free late summer |
| Andaman & Nicobar, Lakshadweep | Jan–Apr | 3% | Island dry season |

### 2.2.3 Reference labels

To judge any automatic labelling method we needed labels we could trust. Each pair was assigned one of four classes: *change* for a lasting, visible land-use change such as new or expanded buildings, roads, construction sites, quarries, mines, new water bodies, cleared vegetation or solar farms; *seasonal* when only crop stage, vegetation greenness or water level differ; *no change*; and *can't tell* when haze or cloud prevents a decision. Change and seasonal pairs also receive a one-sentence description of what differs and where, which is the caption a future dataset would carry.

We built a web-based review tool for this purpose. It works in two steps. In the first, the annotator sees the two dates side by side at full size, together with a blink view that flips between them in place, and chooses one of the four classes before any model output is visible. In the second, the automatic suggestion is shown and, for change and seasonal pairs, the annotator accepts the drafted description or rewrites it. The annotator's answer is always final, and the blind first choice is stored separately so that any influence of the suggestion can be measured. For the evaluation in this paper only the first, blind step was used; the second step is part of the toolkit for building the final dataset.

In total 1,002 pairs carry labels. Of these, 535 were labelled blind by a human annotator. The reviewed pairs were all pairs on which the models tested in [Section 3.3](#sec-labelexp) disagreed with the automatic suggestions, plus a random sample of 50 pairs on which they agreed. The remaining 467 pairs carry the automatic development labels, prepared with model assistance and visual inspection but not reviewed by the annotator. Class counts are given in [Table 2](#tab2), and examples are shown in [Figure 6](#fig6).

Table: []{#tab2}**Table 2.** Label counts.

| Label set | Change | Seasonal | No change | Can't tell | Total |
|---|---|---|---|---|---|
| Human-labelled (blind) | 55 | 214 | 226 | 40 | 535 |
| All labelled pairs | 143 | 400 | 383 | 76 | 1,002 |

# 3. Methodology {#sec-method}

The work was organised in three parts ([Figure 2](#fig2)). Part A developed captioning models on the existing benchmarks and tested their transfer. Part B applied off-the-shelf captioners to Indian imagery. Part C built the Indian dataset and tested whether vision-language models could label it. Each part started from the outcome of the one before, and within each part every experiment changed one component and was scored with the same measures. [Table 3](#tab3) summarises the sequence, the question each step answered and the decision it led to.

![[]{#fig2}**Figure 2.** Workflow of the study. Each part was motivated by the outcome of the previous one.](figures/fig2_workflow.png){width=100%}

Table: []{#tab3}**Table 3.** Experimental sequence, outcome of each step and the reason for the next one.

| Step (2026) | Question | Outcome | Decision taken |
|---|---|---|---|
| A1. LEVIR-CC models (Jul) | Which design captions change best? | The pretrained encoder matters most; fusion adds ≤ 0.009 BLEU-4 | Test transfer before building anything new |
| A2. SECOND-CC transfer (Jul–Aug) | Do these models carry over to another dataset? | Zero-shot BLEU-4 drops 75%; fine-tuning causes forgetting | Try a pretrained language-model decoder |
| A3. Language-model decoders (Aug) | Does a pretrained decoder generalise better? | NaN losses, then degenerate output; one design never trained | Stop training our own captioner; try existing ones |
| B. Off-the-shelf captioners (Sep) | Can existing captioners describe Indian change? | Deny change or produce empty/garbled text | The missing ingredient is data: build an Indian dataset |
| C1. India-wide dataset (Sep) | Can a clean, season-matched dataset be built from free data? | 15,610 pairs; haze is the main obstacle | Build trusted reference labels before automatic labelling |
| C2. Reference labels (Sep–Oct) | What do human labels look like? | Real change is rare (~1 in 7 pairs) | Test whether VLMs can replace the annotator |
| C3. VLM labelling tests (Oct) | Do size, prompts, hints or features make VLM labels reliable? | Each shifts the balance of errors; none removes the seasonal confusion | Validate the best setting at scale |
| C4. 1,000-pair validation (Oct) | Is the best setting precise enough? | At most ~50% precision against the 90% needed | Reject automatic labelling; recommend human annotation |

## 3.1 Captioning models (Part A) {#sec-models}

All Part A models were trained and evaluated in one shared code base, so that only the model changed between experiments. Every model received the same data splits, vocabulary, image preprocessing and loss, and the same optimisation settings: Adam [@adam] with a learning rate of 1×10⁻⁴, weight decay of 1×10⁻⁵, a cosine learning-rate schedule, gradient clipping at 1.0, a batch size of 16 and a fixed random seed of 42. The best checkpoint was chosen by validation loss, and training resumed automatically from the last checkpoint after an interruption. Each model implements the same interface, taking a stacked before–after image pair and a caption and returning word probabilities, and the decoder in most models is a two-layer Transformer [@transformer] with four attention heads.

The models are listed in [Table 4](#tab4). The Phase 1 baseline encodes each image with a small convolutional network of three blocks trained from scratch and combines the two dates as the before features, the after features and their absolute difference. Phase 2 changes only the encoder: both images pass through a frozen RemoteCLIP ViT-B/32 [@remoteclip], a CLIP model [@clip] trained on remote-sensing images and captions, and the same difference fusion feeds the same decoder. Because the trainable capacity of Phase 1 and Phase 2 is almost identical (3.41 M and 3.43 M parameters), any difference between them can be attributed to the encoder.

The next models changed how the two dates are combined. The Final model replaces simple differencing with cross-attention between the before and after features and adds a contrastive loss that pulls each change representation toward the embedding of its own caption ([Eq. 2](#eq2)). Phase 6 splits each image into a 2×2 grid of tiles, encodes each tile with the shared RemoteCLIP backbone, compares tiles in both temporal directions and fuses them with a two-layer Transformer encoder before decoding. Phase 8 stage 1a uses an attention-based difference module, in which the before features attend to the after features, with a light decoder. Phase 8 stage 1b keeps that encoder but passes the change vector through a learned bridge into a frozen Qwen2-0.5B language model [@qwen2], so that a pretrained language model writes the caption.

Phase 7, called CodeAug, was the design aimed most directly at the Indian problem. It used a RemoteCLIP ViT-L/14 encoder with low-rank adapters [@lora] on its feed-forward layers, a Q-Former that compresses the 324 visual tokens of each image to 64 [@blip2], and a 4-bit quantised [@qlora] Qwen2-VL-2B decoder [@qwen2vl]. Against seasonal false alarms it masked vegetation with a green-leaf index computed on both dates and applied the same random colour jitter inside the union of the two masks, so that the model would not learn that a green-to-brown shift means change. It was implemented and shown to fit into 3.48 GB of GPU memory, but it was never trained to a result.

Training minimises the cross-entropy of each reference word given the previous words and the image pair ([Eq. 1](#eq1)). Models with the contrastive term add it with a weight of 0.1. To test transfer, the Phase 6 and Final models trained on LEVIR-CC were fine-tuned sequentially on SECOND-CC at the original learning rate and evaluated on both test sets. [Appendix Table A2](#tabA2) lists the full training settings.

Table: []{#tab4}**Table 4.** Captioning models compared in Part A.

| Model | Encoder / fusion | Decoder | Trainable / total parameters |
|---|---|---|---|
| Phase 1 | CNN from scratch, difference fusion | 2-layer Transformer | 3.41 M |
| Phase 2 | Frozen RemoteCLIP ViT-B/32, difference fusion | 2-layer Transformer | 3.43 M / 154.7 M |
| Final | RemoteCLIP, cross-attention between dates, contrastive loss | 2-layer Transformer | 9.11 M / 160.4 M |
| Phase 6 | RemoteCLIP on 2×2 tiles, tile-fusion Transformer, contrastive loss | 2-layer Transformer | 13.84 M / 165.1 M |
| Phase 8, stage 1a | RemoteCLIP, attention difference module | Light Transformer | 6.29 M / 157.6 M |
| Phase 8, stage 1b | As stage 1a, learned bridge | Frozen Qwen2-0.5B | 5.12 M / 650.4 M |
| Phase 7 (CodeAug, design only) | RemoteCLIP ViT-L/14 + LoRA, Q-Former, vegetation masking | 4-bit Qwen2-VL-2B | 9.81 M / 1.86 B |

[]{#eq1}Captioning loss for a reference caption $w_{1:T}$ given image pair $(I_1, I_2)$:

$$\mathcal{L}_{\text{cap}} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta\left(w_t \mid w_{<t}, I_1, I_2\right)  \text{(1)}$$

[]{#eq2}Contrastive (InfoNCE) alignment between change embeddings $v_i$ and caption embeddings $u_i$ in a batch of size $B$, with temperature $\tau = 0.07$ and weight $\lambda = 0.1$:

$$\mathcal{L}_{\text{con}} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp\left(\cos(v_i,u_i)/\tau\right)}{\sum_{j=1}^{B}\exp\left(\cos(v_i,u_j)/\tau\right)}, \qquad \mathcal{L} = \mathcal{L}_{\text{cap}} + \lambda\,\mathcal{L}_{\text{con}}  \text{(2)}$$

## 3.2 Dataset construction (Part C) {#sec-pipeline}

Each site corresponds to exactly one Sentinel-2 storage block of 1,024×1,024 pixels, or 10.24 km on a side. Sentinel-2 images are stored as cloud-optimised files in blocks of this size, and a read always fetches whole blocks, so a site that straddles block edges can cost four times as much data for the same area. Aligning sites to blocks reduced the volume to be downloaded for the full dataset to roughly 3.4 GB. Downloads ran concurrently with 48 workers, a number chosen after measuring throughput at 12 to 48 workers on our network, and the state of every site was kept in a small SQLite database. A site moved from pending to running to done, no pairs or failed, so an interrupted run, including one stopped by a power cut, resumed where it had stopped instead of starting again. Each saved pair was written to a temporary folder and renamed only when complete, so a crash could not leave half-written files. Transient network failures, which appeared under high concurrency as DNS errors that the raster library cached as missing files, were retried three times with back-off after clearing that cache. Each block was also claimed in the database when saved, so that two nearby sites could never store the same ground twice.

For each site, the catalogue [@earthsearch] was searched in both seasons for scenes whose footprint fully covers the block. The footprint test is inset by 250 m, because footprint polygons drawn in latitude and longitude sag up to about 178 m inside the true tile edge. For the eight least cloudy scenes on each side, the scene-classification layer was read over the block only, which costs about 20 KB per scene, and a scene was kept only if cloud, shadow and missing data covered at most 1% of the block (3% in persistently cloudy regions), snow at most 20% and water at most 70%. A before and an after scene from the same tile were then paired if their positions within the season differed by at most 30 days and their snow cover by at most 5%, so that snowfall would not be mistaken for change. Because both scenes share one tile grid, the block was read directly from each, and the saved pixels are the source pixels, with no reprojection or resampling.

After download, two checks catch what the scene-classification layer misses. Haze lifts the darkest values of the blue band, so a pair was rejected if the darkest-blue levels of the two dates differed by more than 25. A date in which more than 30% of pixels were saturated white was rejected as whiteout. A rejected pair was replaced by the next candidate that did not reuse the faulty date, up to four downloads per site. Each saved pair was then checked for file integrity, for identical pixel grids and for residual misalignment between the dates, which had to stay below half a pixel; 592 patch pairs failed this last check.

Each block was then cut into sixteen patches of 256×256 pixels (2.56 km), and every patch was screened again for missing data, cloud and shadow, snow, water and whiteout. The per-patch limits were 0.1% missing data, 1% cloud or shadow, 20% snow, 70% water and 30% saturated white. Screening, patching and the model-based haze screen ran as a streaming pipeline in batches of about 100 sites, so that one batch could be screened on the cluster while the next was still downloading. Haze that survives these checks is common over the plains in winter, so every remaining pair was rated by Qwen3-VL-8B [@qwen3vl] running in 4-bit precision on the institute's GPU cluster. The model rated haze and cloud from 0 to 3 for each date, and a pair was rejected under the rule in [Eq. 5](#eq5). The 4-bit setting was chosen over 8-bit after a 100-pair trial, because it ran eight times faster on the available 11 GB GPUs (1.4 s per pair).

Two further quantities were stored for every pair. The after image was brightness-matched to the before image with a per-band linear fit ([Eq. 3](#eq3)), and the share of pixels whose mean colour difference after matching exceeds 40 levels was recorded as the pixel-change fraction ([Eq. 4](#eq4)). A pair with a fraction of at least 0.05 was flagged as pixel-changed. Finally, the data were split into training, validation and test sets by site, so that neighbouring patches from the same 10 km block can never fall in two splits. Each pair is released as PNG and GeoTIFF with its metadata.

[]{#eq3}Per-band brightness matching of the after image $A$ to the before image $B$ over valid pixels (least squares):

$$\hat{A}_b = g_b A_b + o_b, \qquad (g_b, o_b) = \arg\min_{g,o}\sum_{i}\left(g A_{i,b} + o - B_{i,b}\right)^2  \text{(3)}$$

[]{#eq4}Pixel-change fraction of a patch with $N$ pixels (a pair is flagged as pixel-changed when $f \ge 0.05$):

$$f = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\left[\frac{1}{3}\sum_{b=1}^{3}\left|\hat{A}_{i,b} - B_{i,b}\right| > 40\right]  \text{(4)}$$

[]{#eq5}Haze screen decision with model ratings $h, c \in \{0,1,2,3\}$ (haze, cloud) for each date:

$$\text{reject} \iff \max\left(h_{\text{before}}, h_{\text{after}}\right) \ge 2 \;\lor\; \max\left(c_{\text{before}}, c_{\text{after}}\right) \ge 2  \text{(5)}$$

## 3.3 Automatic labelling experiments (Parts B and C) {#sec-labelexp}

In Part B, two pretrained captioners were applied without any training to a 30-pair sample of the pilot collection: TEOChat [@teochat], a vision-language assistant trained on temporal Earth-observation data, and Qwen2.5-VL [@qwen25vl], a general vision-language model. Both were run in full precision and in quantised form.

In Part C, every test showed the model the before and after patches, stretched together to the same brightness range and enlarged to 512×512 pixels, at a sampling temperature of 0. Open-weight models were reached through a hosted API, because the larger ones did not fit on the cluster GPUs. Five groups of experiments were run, in the order below; the reason for each next step is given with its result in [Section 4.5](#sec-vlm).

The first compared model size. Qwen3-VL [@qwen3vl] at 8B, 32B and 235B parameters and Gemma 3 at 27B [@gemma3] classified 57 pairs into land-use change, seasonal, no change or haze. The prompt contained a tie-break instruction to choose seasonal when unsure between land-use and seasonal change, and the 12 change pairs in this set came partly from sites with known change. On the same 57 pairs, a third image outlining the changed pixels was added to test whether a plain change outline helps.

The second tested prompt wording and region hints with Qwen3-VL-8B on 130 pairs (33 change, 82 seasonal or no change, and 15 controls). The neutral baseline used a four-class prompt without a tie-break rule. Boxes around the three most-changed areas were then drawn on both images, taken either from pixel differences or from distances between ResNet-50 [@resnet] feature maps, with and without enlarged crops of each box. A further variant told the model that change had been detected inside the boxes and asked it to describe that change. To measure invented change directly, the 15 control pairs showed the same before image twice. The boxed variants used a per-box prompt, so the comparison between the baseline and the boxed variants combines two effects; the pixel-box and CNN-box variants share one prompt and can be compared directly.

The third tested a rule-based change map proposed by a teammate. It classifies each date into vegetation, bare soil, water, building, road and shadow, colours every from–to transition and passes the map to the model together with a prompt that tells it to treat the map as fact. The method was designed for sub-metre imagery; we ran it unchanged except for setting its road width to 2 pixels (about 20 m), as its own documentation prescribes for 10 m data, and evaluated it on 50 pairs with Qwen3-VL-32B.

The fourth asked whether simple image features could decide without any language model. Colour difference after brightness matching, newly appearing edges, pixels that became bright and grey, the size of the largest changed region and the ResNet-50 feature distance were combined in a classifier trained on the labelled pairs and evaluated with cross-validation grouped by site.

The fifth group concerned the final decision rule. A stricter two-model rule was first tried on a separate 100-pair sample: a pair counted as changed only if the 32B model said so with a stated confidence of at least 0.9, the 8B model agreed, and the image zone named by the model actually contained changed pixels. A free-form question with no categories and no output format was then tried on 37 decidable pairs with the 32B and 235B models: "Before and after images of the same place, 6 years apart. Has anything lasting changed (like buildings, roads, ponds, cleared land), or are the differences only seasonal?" The models answered in their own words, and a separate text-only call classified each answer as land-use change, seasonal, no change or unclear. This plain question was finally put to both models for all 1,002 labelled pairs, and four decision rules were compared: each model alone, both models agreeing, and either model.

## 3.4 Evaluation {#sec-eval}

Captions were scored with BLEU-1 to BLEU-4 [@bleu] ([Eq. 6](#eq6)), METEOR [@meteor], ROUGE-L [@rouge] ([Eq. 7](#eq7)) and the cosine similarity between Sentence-BERT embeddings of the predicted and reference captions [@sbert] ([Eq. 8](#eq8)), using the all-MiniLM-L6-v2 encoder. BLEU counts matching word sequences, METEOR also credits stems and synonyms, ROUGE-L measures the longest common subsequence, and the embedding similarity rewards captions that mean the same thing in different words. Our pipeline also computed a simplified CIDEr-like score, but it is not comparable to CIDEr as published [@cider], so we do not report it. The seasonal false-positive and structural false-negative rates planned for CodeAug were defined but never computed, because that model never trained.

For the labelling experiments, a "change" call is a positive. We report recall (the share of real changes found), precision (the share of change calls that are correct) and the false-change rate (the share of unchanged pairs called changed), defined in [Eq. 9](#eq9), with 95% Wilson score intervals [@wilson] ([Eq. 10](#eq10)). Precision is the decisive measure for automatic labelling. Real change is rare in an evenly sampled dataset, so even a small false-change rate produces many false labels, and every false "change" label teaches a later model to describe something that is not there. We set 90% precision as the minimum for a usable automatic labeller.

[]{#eq6}

$$\text{BLEU-}N = \text{BP}\cdot\exp\left(\sum_{n=1}^{N}\frac{1}{N}\log p_n\right), \qquad \text{BP} = \min\left(1, e^{\,1 - r/c}\right)  \text{(6)}$$

where $p_n$ is the modified $n$-gram precision, $c$ the candidate length and $r$ the reference length.

[]{#eq7}

$$R_{\text{lcs}} = \frac{\text{LCS}(X,Y)}{|X|}, \quad P_{\text{lcs}} = \frac{\text{LCS}(X,Y)}{|Y|}, \quad F_{\text{lcs}} = \frac{(1+\beta^2)R_{\text{lcs}}P_{\text{lcs}}}{R_{\text{lcs}} + \beta^2 P_{\text{lcs}}}  \text{(7)}$$

[]{#eq8}

$$\text{cos}(e_{\text{pred}}, e_{\text{ref}}) = \frac{e_{\text{pred}}\cdot e_{\text{ref}}}{\lVert e_{\text{pred}}\rVert\,\lVert e_{\text{ref}}\rVert}  \text{(8)}$$

[]{#eq9}

$$\text{Precision} = \frac{TP}{TP+FP}, \qquad \text{Recall} = \frac{TP}{TP+FN}, \qquad \text{FPR} = \frac{FP}{FP+TN}  \text{(9)}$$

[]{#eq10}

$$\frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}, \qquad z = 1.96  \text{(10)}$$

# 4. Results and discussion {#sec-results}

## 4.1 Captioning on LEVIR-CC {#sec-levir}

[Table 5](#tab5) and [Figure 3a](#fig3) show the LEVIR-CC test results. The clearest effect in the whole of Part A is the change of encoder. Moving from the CNN trained from scratch (Phase 1) to the frozen RemoteCLIP encoder (Phase 2) improved every metric: BLEU-4 rose from 0.303 to 0.328 (8.5% relative), METEOR from 0.597 to 0.630, ROUGE-L from 0.621 to 0.655 and the embedding similarity from 0.648 to 0.706, while test loss fell by 3.6%. Since the two models have almost the same number of trainable parameters and differ only in the encoder, the gain comes from features learned on remote-sensing imagery rather than from extra capacity.

Everything we added after that helped little. Cross-attention between dates with contrastive alignment (Final) and the tile-based design (Phase 6) reached BLEU-4 of 0.337 and 0.336, at most 0.009 above Phase 2, while using up to four times as many trainable parameters. Phase 8 stage 1a, with its attention-based difference module, scored 0.325, slightly below Phase 2. With a single training run per model we cannot claim that the small differences between Final and Phase 6 are real. LEVIR-CC rewards a fairly short list of sentence patterns, and once the encoder supplies good features, a better fusion module has little left to improve. The LEVIR-CC authors already used a pretrained encoder in their own baseline [@levircc], which fits our finding that the encoder is the decisive component.

The training curves point to the same limit. Validation loss reached its minimum at epoch 6 for Phase 2 ([Figure 3b](#fig3)) and at epoch 7 for Phase 8 stage 1a, while training loss kept falling; by epoch 15, Phase 8 stage 1a had a training loss of 0.48 and a training token accuracy of 85% against a validation loss of 1.14 ([Appendix Table A8](#tabA8), [Appendix Figure A1](#figA1)). Early stopping on validation loss was therefore essential. The metrics also respond differently to the encoder change. The embedding similarity gained the most (9.0% relative) and METEOR and ROUGE-L the least (about 5%), which suggests that the RemoteCLIP features mostly helped the decoder choose the right kind of change, such as buildings rather than roads, rather than produce longer exact matches with the reference wording.

The saved test predictions show what the scores hide ([Figure 4](#fig4), [Appendix Table A4](#tabA4)). On 300 test samples, whose references contain 163 different sentences, the Phase 6 and Phase 8 runs produced only 21 and 32 distinct captions. They often recognised that something had been built, but described it with a stock phrase such as "a huge residential area is built in almost whole scene", even when the reference said that a single house had appeared in a corner ([Appendix Figure A2](#figA2)). Both runs also missed changes: they captioned 43 and 31 of the 164 changed pairs as "there is no difference", and on one unchanged pair the Phase 6 run reported vegetation loss. The models learned the common templates of the benchmark well and the specific content of each scene poorly.

*Why the next step.* A pretrained remote-sensing encoder was clearly the right foundation, but LEVIR-CC covers planned suburbs in a single region, and our target was Indian imagery. Before building anything new, we needed to know whether these models transfer to another dataset at all, so we evaluated them on SECOND-CC and fine-tuned them on it.

Table: []{#tab5}**Table 5.** LEVIR-CC test results (n = 1,929). Losses are comparable only within the same vocabulary (Phases 1, 2, 8: 733 words; Final, Phase 6: 1,300 words). Full BLEU-1 to BLEU-4 scores are in [Appendix Table A1](#tabA1).

| Model | Test loss | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | Cosine |
|---|---|---|---|---|---|---|
| Phase 1 | 0.969 | 0.472 | 0.303 | 0.597 | 0.621 | 0.648 |
| Phase 2 | 0.935 | 0.517 | 0.328 | 0.630 | 0.655 | 0.706 |
| Final | **0.898** | **0.531** | **0.337** | **0.639** | 0.660 | 0.712 |
| Phase 6 | 0.923 | 0.521 | 0.336 | 0.636 | **0.662** | **0.715** |
| Phase 8, stage 1a | — | 0.509 | 0.325 | 0.616 | 0.648 | 0.701 |
| Phase 8, stage 1b | — | 0 | 0 | 0 | 0 | 0.010 |
| Phase 7 (CodeAug) | no valid result | | | | | |

![[]{#fig3}**Figure 3.** Part A results. (a) LEVIR-CC test scores per model; (b) training and validation loss of the Phase 2 model; (c) LEVIR-CC scores of the Phase 6 and Final models before and after fine-tuning on SECOND-CC.](figures/fig3_partA_results.png){width=100%}

![[]{#fig4}**Figure 4.** Prediction behaviour on 300 saved LEVIR-CC test samples. (a) Share of changed pairs captioned "there is no difference"; (b) number of distinct captions produced.](figures/fig4_prediction_behaviour.png){width=100%}

## 4.2 Domain transfer and language-model decoders {#sec-transfer}

Moving to a second dataset exposed how narrow these models are ([Table 6](#tab6)). Without any SECOND-CC training, Phase 8 stage 1a reached a BLEU-4 of only 0.081 on SECOND-CC, about 75% below its LEVIR-CC score. Fine-tuning on SECOND-CC raised BLEU-4 there to 0.123 for Phase 6 and 0.130 for the Final model, but at a heavy cost on LEVIR-CC: BLEU-4 fell from 0.336 to 0.121 and from 0.337 to 0.149, a loss of 56–64%, and test loss rose from about 0.9 to 3.5–4.1 ([Figure 3c](#fig3), [Appendix Table A3](#tabA3)). METEOR and ROUGE-L fell less than BLEU, so general fluency survived while the LEVIR-specific phrasing was lost. This is the classic pattern of catastrophic forgetting, in which training on a new task overwrites the weights that served the old one [@kirkpatrick].

The prediction sheets show the extreme case. After fine-tuning, the Final model answered "there is no difference" for every one of the 300 saved LEVIR-CC samples, including all 164 changed pairs ([Figure 4a](#fig4)). The SECOND-CC validation loss had reached its lowest point, 1.522, at epoch 6 and then crept upward to 1.551 by epoch 14 while training continued, which suggests that fine-tuning ran too long at the original learning rate. A shorter, lower-rate configuration (learning rate 2×10⁻⁵, weight decay 1×10⁻⁴, six epochs) was defined afterwards but not evaluated. Results on SECOND-CC itself remained far below LEVIR-CC levels in any case ([Appendix Table A9](#tabA9), [Appendix Figures A3](#figA3) and [A10](#figA10)).

*Why the next step.* Catastrophic forgetting suggested that a small decoder trained from scratch has too little general language knowledge to adapt. A pretrained language model might generalise better, so we replaced the decoder with Qwen-family language models.

Replacing the small decoder with a pretrained language model, which in principle brings broad knowledge of language, did not work on our hardware. In Phase 8 stage 1b the training loss first diverged to NaN. The cause was traced to the bridge between the visual features and the frozen Qwen2-0.5B model: without normalisation, its output could exceed the range of 16-bit floating point (about 65,504) inside the roughly 24 frozen language-model layers. Adding a layer normalisation fixed the loss, and training then reached a best validation loss of 1.42 at epoch 4. Yet every n-gram metric on both test sets was exactly zero; the model's free-running output was degenerate even though its loss under teacher forcing looked reasonable. A plausible teacher-forced loss is therefore no evidence of usable generation. The CodeAug design was shown to fit in 3.48 GB of GPU memory but never completed a training run, and the planned comparison of the two language-model routes failed on a constructor error before it started. Patch-wise inference on 43 unlabelled high-resolution pairs also made the output more uniform rather than more specific: whole-image inference produced 23 distinct captions, mostly "vegetation quantity is decreased", whereas patch-wise inference produced 9, and one of them, "the green region is replaced by a huge house at the bottom right", was given for 75% of the pairs regardless of where the change was ([Appendix Figure A4](#figA4)).

*Why the next step.* Training our own captioner had stalled. The remaining question was whether existing, already-trained vision-language captioners could describe Indian change directly, with no training on our side.

Table: []{#tab6}**Table 6.** SECOND-CC test results (n = 1,227) and LEVIR-CC BLEU-4 after fine-tuning. Zero-shot and fine-tuned rows come from different models.

| Condition | BLEU-4 | METEOR | ROUGE-L | Cosine | LEVIR-CC BLEU-4 after |
|---|---|---|---|---|---|
| Phase 8 (1a), zero-shot | 0.081 | 0.298 | 0.300 | 0.378 | — |
| Phase 6, fine-tuned | 0.123 | 0.392 | 0.416 | 0.530 | 0.336 → 0.121 |
| Final, fine-tuned | 0.130 | 0.413 | 0.433 | 0.550 | 0.337 → 0.149 |

## 4.3 Off-the-shelf captioners on Indian imagery {#sec-offshelf}

With our own models stuck, we asked whether existing captioners could describe Indian change directly. They could not ([Table 7](#tab7)). TEOChat answered "Nothing meaningful changed" for 27 of the 30 pilot pairs, and for 23 of 30 when run in 4-bit precision, although the pilot tiles had been selected precisely because they showed strong change. One of its few other answers described damage from a hurricane that had not happened. Qwen2.5-VL failed in the opposite way. On its first run it returned 11 empty answers and 12 garbled ones, made of repeated tokens or text in another language, out of 30. Retries and 8-bit loading reduced these failures without removing them, and when the model did answer it tended to tell a confident, generic story of urban growth about scenes that were largely unchanged ([Appendix Figures A5](#figA5) and [A6](#figA6)).

Table: []{#tab7}**Table 7.** Off-the-shelf captioners on 30 pilot Indian Sentinel-2 pairs.

| Captioner | Setting | Outcome |
|---|---|---|
| TEOChat | Full precision | "Nothing meaningful changed" on 27/30; one invented hurricane-damage description |
| TEOChat | 4-bit | "Nothing meaningful changed" on 23/30 |
| Qwen2.5-VL | First run | 11 empty and 12 garbled answers (repeated tokens, non-English text) |
| Qwen2.5-VL | Retries, 8-bit | Fewer failures; answers describe generic urban development |

These two failures, denying change everywhere and inventing it everywhere, reappear in sharper form in the Part C experiments.

*Why the next step.* No available model, ours or off-the-shelf, produced usable change captions for Indian imagery, and there was no labelled Indian data on which any model could be trained or fairly judged. The common missing ingredient was data, so we built a large, quality-controlled Indian bi-temporal dataset, planning to label it with stronger vision-language models.

## 4.4 India-wide dataset {#sec-dataset}

The pipeline examined 27,888 patch pairs and accepted 15,610 of them, from 1,669 sites in 35 states and union territories; Lakshadweep is absent because its sites are open sea. The split by site gives 12,303 training, 1,706 validation and 1,601 test pairs. Of the 12,278 rejected pairs, 10,994 (90%) were rejected by the haze and cloud screen, far more than for any other reason ([Figure 5a](#fig5)). Misalignment (592), water (408), snow (133), missing data (77), residual cloud (55) and whiteout (19) together removed fewer than 1,300 pairs. Haze, not cloud, is the main obstacle to clean bi-temporal Sentinel-2 imagery over India, especially over the northern plains in winter; the regional breakdown is given in [Appendix Table A5](#tabA5). Acceptance rates differ sharply by region. About 66% of the north-eastern patches were accepted, probably because cloudy scenes had already been filtered out at the scene level, against 59% for the Deccan, 55% for the northern plains, 51% for the western desert and only 40% for the southern states and islands, where water and haze removed many patches. Moving the plains sites to a February–April window, after the worst of the winter smog, recovered pairs for 193 of 195 of them.

The haze screen itself is imperfect ([Figure 5b](#fig5)). In the 100-pair trial, of which 84 pairs could be decided, the 4-bit model we used caught 81% of hazy pairs but also rejected 27% of clean ones. Running the same 8B model at 8-bit rejected fewer clean pairs (12%) but caught fewer hazy ones (70%), and it ran far slower. The 235B model performed best, at about 92% accuracy, but that figure is an upper bound: the trial labels were re-checked with its help, which favours it. Some thin haze therefore remains among the accepted pairs, and some clean pairs were lost.

*Why the next step.* Before labelling 15,610 pairs automatically, we needed trustworthy reference labels against which to measure any labelling method, so we built the review tool and the human-labelled reference set described in [Section 2.2.3](#sec-datasets).

[Figure 6](#fig6) shows one human-labelled example of each class. Seen side by side, the difficulty is plain. The seasonal and no-change pairs both contain strong colour differences between dates, while the change pair shows a real but small alteration next to a reservoir whose water level has also changed. Among the labelled pairs, only about one in seven shows lasting land-use change. Pixel-level change, by contrast, is common, because the pixel-change flag of [Eq. 4](#eq4) is mostly triggered by crop cycles. The blind human label matched the automatic suggestion on 84% of the reviewed pairs, so the suggestions were wrong about one time in six even before any model was tested.

![[]{#fig5}**Figure 5.** Dataset screening. (a) Rejected patch pairs by reason (12,278 of 27,888); (b) haze-screen trial on 84 decidable pairs for four model settings. The 235B row is an upper bound because the trial labels were checked with its help.](figures/fig5_screening.png){width=100%}

![[]{#fig6}**Figure 6.** Human-labelled examples (left: before, 2019–20; right: after, 2025–26). (a) Change; (b) seasonal; (c) no change.](figures/fig6_class_examples.png){width=70%}

*Why the next step.* With a reference set in place, we could test whether vision-language models label change accurately enough to replace the human annotator.

## 4.5 Can vision-language models label land-use change? {#sec-vlm}

The labelling experiments ask whether the dataset could be labelled automatically. The short answer is no, and each experiment shows a different side of why. [Table 8](#tab8) collects the results of all the development experiments; [Table 9](#tab9) gives the final validation.

Table: []{#tab8}**Table 8.** Development experiments on automatic change labelling. "Changes found" counts real change pairs called change; "false changes" counts seasonal or unchanged pairs called change; controls are pairs in which the same image is shown twice.

| Experiment | Model | Changes found | False changes | False on controls |
|---|---|---|---|---|
| 4-class prompt with tie-break (57 pairs) | Qwen3-VL-8B | 10/12 | 8/45 | — |
| | Gemma 3 27B | 8/12 | 16/45 | — |
| | Qwen3-VL-32B | 11/12 | 16/45 | — |
| | Qwen3-VL-235B | 4/12 | 1/45 | — |
| Same, with change-outline image | Qwen3-VL-8B | 6/12 | — | — |
| Neutral 4-class prompt (130 pairs) | Qwen3-VL-8B | 24/33 | 12/82 | 0/15 |
| Pixel-difference boxes | Qwen3-VL-8B | 11/33 | 3/82 | 0/15 |
| Pixel boxes + enlarged crops | Qwen3-VL-8B | 14/33 | 7/82 | 0/15 |
| CNN feature boxes | Qwen3-VL-8B | 10/33 | 6/82 | 0/15 |
| CNN boxes + enlarged crops | Qwen3-VL-8B | 16/33 | 4/82 | 0/15 |
| Boxes with forcing prompt | Qwen3-VL-8B | 29/33 | 51/82 | 12/15 |
| Rule-based change map as fact (46 decidable) | Qwen3-VL-32B | 19/19 | 27/27 | — |
| Strict two-model triage (100 pairs) | 32B + 8B | 7/15 | — | — |
| Free-form question (37 pairs) | Qwen3-VL-32B | 10/10 | 13/27 | — |
| | Qwen3-VL-235B | 8/10 | 4/27 | — |

**Model size.** Larger models were not better at finding change; they were more or less willing to call it ([Figure 7a](#fig7)). On the 57-pair set, Qwen3-VL-32B found 11 of 12 changes but also called 16 of 45 unchanged pairs changed. Gemma 3 27B behaved similarly (8 of 12, with 16 false calls). Qwen3-VL-235B was the most cautious, raising only one false change but finding only 4 of the 12. The 8B model sat in between, at 10 of 12 with 8 false calls, and was the most sensitive to prompt wording. The tie-break instruction in this prompt pushes models toward "seasonal", and the most conservative model followed it most closely, so model size shifted the balance between missed and false changes without removing the underlying confusion. Adding a third image that outlined the changed pixels did not help any model; the 8B model's detections fell from 10 to 6 of 12.

*Why the next step.* If the models could not find change unaided, perhaps they could confirm it when shown where to look, so we tested region hints.

**Region hints.** They did not help ([Figure 7b](#fig7)). With the neutral four-class prompt and no hints, the 8B model found 24 of 33 changes, with 12 false calls among 82 unchanged pairs and none on the identical-image controls. With neutral boxes from pixel differences or CNN features it found only 10 to 16. In these runs the model's overall answer simply repeated its answers for the boxes, and it never reported a change outside them, even when asked to. Because the boxes are drawn where pixels changed most, they usually landed on fields whose colour had changed with the season rather than on the new structures ([Appendix Figure A7](#figA7)). The no-hint baseline used a different prompt from the boxed variants, so part of this drop may come from the prompt; the two box sources, which share a prompt, performed alike, and enlarged crops recovered only a few detections. Telling the model that change had been detected in the boxes had the opposite effect. It then found 29 of 33 changes, but also called 51 of 82 unchanged pairs changed and invented change in 12 of the 15 control pairs in which the two images were identical.

*Why the next step.* A richer hint, a classified change map with from–to transitions, might succeed where boxes failed, so we tested the map proposed by a teammate.

**Rule-based change map.** The change map marked a median of 82% of each image as changed (maximum 98%), often labelling crop and soil transitions as buildings or roads, and the model, told to trust the map, reported change on every one of the 46 decidable pairs: all 19 changed pairs and all 27 unchanged ones ([Appendix Figure A8](#figA8)). At 10 m resolution, a colour-based change map describes the season, and presenting it as fact makes the model describe the season as construction. Change-Agent [@changeagent] couples change captioning with a change-detection branch trained for the purpose, whereas our hints came from untrained pixel differences and generic CNN features; whether better-localised hints would help remains untested. In four separate tests (the outline image, the region boxes, the forcing prompt and the change map), visual hints either narrowed the model's attention or made it invent change.

*Why the next step.* Since visual prompting failed, we tested whether image features alone could at least filter pairs before any model call.

**Feature gate.** Simple image features ranked pairs reasonably but could not decide. The best combination reached a precision–recall area of 0.45, against 0.16–0.20 for colour difference alone, yet at 90% recall it still passed about 62% of all pairs ([Appendix Figure A9](#figA9)). Such a gate is useful for ordering pairs for review, not for labelling them.

**Strict triage and prompt wording.** The strict two-model rule found only 7 of 15 changes in its 100-pair sample, and the stated confidence on which it relied turned out to be uninformative: the 8B model reported a confidence of 0.95 for every one of the 60 labelled pairs in that sample. Asking for a fixed answer format also hid the reasoning behind each answer. With the free-form question, the 32B model found all 10 changes among 37 pairs with 13 false calls, and the 235B model found 8 with 4 false calls. The models explained what they saw in their own words, which made each error easy to trace to its cause, usually a field or water body read as construction or a construction site read as farmland; the 235B model in particular tended to describe real construction as "seasonal or agricultural variation". Wording shifted the balance between missed and false changes in every version we tried, but no wording removed the confusion between seasonal and lasting change.

*Why the next step.* The plain, unassisted question was the most promising setting so far, so we validated it at scale against the human labels with the two strongest models.

**1,000-pair validation.** The full run over 1,002 pairs cost USD 1.00 in API fees, so cost was not the obstacle ([Table 9](#tab9), [Figure 8](#fig8)). On the 495 decidable human-labelled pairs, Qwen3-VL-32B found 89% of real changes, but only 11% of its change calls were correct, because it called 87% of unchanged pairs changed, including 200 of the 214 seasonal pairs ([Appendix Table A6](#tabA6)). Qwen3-VL-235B was far more cautious, finding 45% of changes with 17% precision. Requiring both models to agree gave 18% precision. These pairs were deliberately the hard cases, since they were chosen for review where the models and the automatic labels disagreed, so these precision values are pessimistic. Across all 962 decidable labelled pairs, which include the easier ones, the agreement rule reached 51% precision with 78% recall ([Appendix Table A7](#tabA7)). Both views lead to the same conclusion: at best about half of the model's "change" labels would be right, against the 90% or more that an automatically built dataset needs.

The free answers make the errors easy to read ([Figure 9](#fig9)). For a genuine change the 235B model wrote that "there has been significant, lasting development, including new buildings, roads, and land clearing". For a seasonal pair it claimed "new agricultural plots, possible expansion of settlement, and infrastructure development". For a real change next to farmland it concluded that the differences were "consistent with seasonal or annual agricultural cycles". The same model tells the same kind of story in all three cases; what it cannot do reliably is decide which story fits the scene.

The low precision follows from the rarity of change. In our labelled pairs, unchanged pairs outnumber changed ones by about six to one. A rule that wrongly flags only 13% of unchanged pairs therefore produces almost as many false change labels as true ones, which is exactly what we observed. This also means that the human review step cannot be replaced by asking models to check each other: in a random sample of 45 pairs on which the automatic labels and both models agreed, the human disagreed on 9, missing 4 real changes and accepting 5 false ones.

Taken together, the experiments suggest that the problem is not one of prompting skill or model size but of evidence. A true-colour pair at 10 m often does not contain enough information to separate a newly ploughed field from a cleared construction plot, or a filled reservoir from a new one, and a model trained mostly on photographs and sharper imagery fills that gap with the most common story it knows. People face the same ambiguity, which is why our annotator used a blink view and could still mark 40 pairs as undecidable. Any automatic route will need additional evidence, such as extra bands, a third date or sharper imagery, rather than a better way of asking the same question. This closes the experimental sequence: automatic labelling is rejected as a source for the dataset, and the evidence points to human annotation.

![[]{#fig7}**Figure 7.** Vision-language labelling tests. (a) Model size with a 4-class prompt (57 pairs); (b) Qwen3-VL-8B with and without region hints and with a forcing prompt (130 pairs and 15 identical-image controls).](figures/fig7_vlm_tests.png){width=100%}

Table: []{#tab9}**Table 9.** 1,000-pair validation on the human-labelled pairs (n = 495; 55 change), with 95% Wilson intervals.

| Rule | Recall | Precision | False-change rate | Accuracy |
|---|---|---|---|---|
| Qwen3-VL-32B | 89% (78–95) | 11% (9–15) | 87% (84–90) | 21% |
| Qwen3-VL-235B | 45% (33–58) | 17% (12–24) | 27% (23–32) | 70% |
| Both agree | 42% (30–55) | 18% (12–25) | 24% (21–29) | 72% |
| Either | 93% (83–97) | 11% (9–15) | 90% (87–93) | 19% |

![[]{#fig8}**Figure 8.** Recall and precision of "change" calls for each decision rule. (a) Human-labelled pairs; (b) all labelled pairs. The dashed line marks the 90% precision needed for automatic labelling.](figures/fig8_validation.png){width=100%}

![[]{#fig9}**Figure 9.** Validation examples with the human label as reference. (a) Correct detection; (b) false alarm on a seasonal pair; (c) missed change.](figures/fig9_validation_examples.png){width=60%}

# 5. Limitations and future scope {#sec-limits}

Several limitations qualify these results. Each Part A model was trained once with a single seed, so small differences between models are not statistically supported. The CIDEr-like score in our pipeline was a simplification and was left out, and the seasonal false-positive measures planned for the CodeAug design were never computed. Both language-model designs remain unvalidated: one produced degenerate output and the other never trained. Test losses are only comparable between models that share a vocabulary: Phases 1, 2 and 8 use 733 words, the Final and Phase 6 models 1,300 words, and Phase 8 stage 1b the Qwen tokenizer.

The Indian experiments have their own limits. At 10 m a single house covers one or two pixels, so much real change, such as individual buildings, narrow roads and small plots, is at or below what anyone can see, human or model. The human labels come from one annotator and are weighted toward hard cases, and the labels on the 467 unreviewed pairs are estimated, from a random sample, to be wrong about one time in five. The 4-bit haze screen passed about 19% of hazy pairs and rejected about 27% of clean ones in its trial, so some accepted pairs still contain haze. The prompting experiments used 40 to 135 pairs each, so their figures are indicative rather than precise. Free answers were mapped to classes by a text-only Qwen3-VL-32B call, a model from the same family as the ones being tested, although spot checks agreed with the answers' stated conclusions.

Finally, compute shaped what could be tried ([Table 10](#tab10)). The institute's cluster offered one older 11 GB GPU per account (GTX 1080 Ti or RTX 2080 Ti) without bfloat16 support. This forced 4-bit models and very small batches, caused numerical failures in the language-model designs, and could run only the 8B model, which in every test was weaker than the larger hosted models. Jobs waited about 30 minutes in the queue, models had to be downloaded again on each new node, four nodes had faulty GPUs, storage was not available on the GPU nodes, and the cluster was unreachable for days during the final experiments. All decisive Part C experiments were therefore run on hosted models.

Table: []{#tab10}**Table 10.** Engineering issues and their effect on the study.

| Issue | Effect | Resolution |
|---|---|---|
| Cluster GPUs: 11 GB, no bfloat16, one per account | 4-bit models, tiny batches, CPU fallback (Phase 8 stage 1a ran on CPU) | Quantisation, gradient accumulation; hosted models for decisive experiments |
| PyTorch support for older GPUs | Mixed software stacks across runs | Version pinning, partly inconsistent |
| fp16 overflow in the language-model bridge | NaN loss | Layer normalisation on the bridge output |
| Degenerate generation in Phase 8 stage 1b | All n-gram metrics zero | Not resolved |
| Forgetting during fine-tuning | LEVIR-CC performance collapsed | Lower-rate configuration defined, not evaluated |
| Cluster unreachable (Sep–Oct) | Local GPU experiments impossible | Hosted API models used instead |

The work also points to several clear next steps:

1. **Distillation from a stronger captioner.** If a future model can caption the 15,610 pairs reliably, verified against the human-labelled set at 90% precision or better, its captions can be used to teacher-force a small captioner such as the Phase 2 model, which has only 3.4 M trainable parameters. The large model would then be needed only once, offline, as a labeller, and the small model could run cheaply on modest hardware.
2. **Human-in-the-loop labelling.** The feature gate and model answers can be used to rank pairs, and annotators can review the ranked pairs with the existing two-step review tool, which asks for a blind category first and a caption check second. Active learning can then direct effort to the pairs a model is least sure about.
3. **Change-enriched sampling.** Real change is rare at evenly spread sites. Sampling around places where change is likely, such as expanding town edges, new highways, mines, construction and reservoir projects found through OpenStreetMap, would raise the share of changed pairs from about 15% to an estimated 50%.
4. **More evidence per pair.** Near-infrared and short-wave infrared bands, and a third, intermediate date, would help separate lasting change from crop cycles and water-level variation. Sharper imagery, where available, would make individual buildings visible.
5. **Trained change detectors for localisation.** Detectors trained for change, of the kind used in Change-Agent [@changeagent], could propose regions, provided the regions are used only to locate and describe a change that has already been confirmed, never to decide whether change occurred.
6. **Adaptation without forgetting.** Domain transfer should be repeated with the lower-rate configuration, replay of source data, or adapter-only training [@lora], and the language-model decoders revisited once their generation fault is fixed.
7. **Rigour.** Future runs should use several seeds with confidence intervals, the standard CIDEr metric [@cider] and dedicated measures of seasonal false positives and structural false negatives.
8. **Recovering clean pairs.** The 10,994 pairs rejected by the 4-bit haze screen can be re-screened with a stronger model to recover the clean pairs it rejected.

# 6. Conclusions {#sec-concl}

This study set out to make change captioning work for Indian Sentinel-2 imagery and tested, step by step, the shortcuts that seemed most likely to get there.

On the existing benchmarks (O1), a frozen encoder pretrained on remote-sensing imagery was the single most useful choice, raising BLEU-4 on LEVIR-CC from 0.303 to 0.328 at the same trainable capacity, while more elaborate fusion added at most 0.009. The models learned the benchmark's templates rather than the content of each scene, and transfer to a second dataset by fine-tuning erased most of what they had learned; in one case the output collapsed to a single caption. Language-model decoders could not be trained to a working state on the available hardware.

Existing captioners applied directly to Indian imagery (O2) were unusable: one denied change almost everywhere and the other produced empty, garbled or invented descriptions.

We then built an India-wide dataset of 15,610 quality-screened, season-matched Sentinel-2 pairs from 1,669 sites (O3). Haze was by far the largest quality problem, and real land-use change turned out to be rare at evenly sampled sites.

Finally (O4), no combination of model size, prompt, region hint, change map or decision rule produced change labels accurate enough to build the dataset automatically. The best rule was correct on at most about half of its change calls, because every model confused seasonal crop and water differences with lasting change, and hints either narrowed the model's attention or made it invent change.

Neither transfer from existing datasets nor automatic labelling with current vision-language models gives reliable change captions for 10 m Indian imagery, and scaling either route would produce a large dataset of poor quality. The screened dataset, the 1,002 labelled pairs, the review tools and the evaluation protocol released with this work are a sound base for the human-annotated dataset that the task still requires.

# Acknowledgements {#sec-ack}

We thank our mentor, Rama Chandra Prasad, for his guidance throughout this study. Sentinel-2 data were provided by the European Union's [Copernicus programme](https://dataspace.copernicus.eu) and the European Space Agency; this work contains modified Copernicus Sentinel data (2019–2026). We thank Element 84 and the [AWS Open Data programme](https://registry.opendata.aws/sentinel-2-l2a-cogs/) for the Earth Search catalogue, [geoBoundaries](https://www.geoboundaries.org) for the administrative boundaries, the authors of LEVIR-CC and SECOND-CC for releasing their datasets, and the developers of RemoteCLIP, Qwen and Gemma for making their models available. Model inference used the [OpenRouter](https://openrouter.ai) platform, training and screening used the institute's Ada high-performance computing cluster, and the dataset and models are hosted on [Kaggle](https://www.kaggle.com/kspsvlnsiddardha) and [Hugging Face](https://huggingface.co/Kspsvln).

# Data and code availability {#sec-avail}

The code, the India-wide dataset, the labelled subset, the prediction sheets and the model checkpoints are listed with direct links in [Appendix B](#appB): code and branches in [Table B1](#tabB1), checkpoints per model in [Table B2](#tabB2), released datasets in [Table B3](#tabB3), and third-party datasets, models and services in [Table B4](#tabB4).

# References {#sec-refs}

[[REFERENCES]]

# Appendix A: Supplementary tables and figures {#appA}

Appendix tables and figures are cited in the main text as Appendix Table A*n* and Appendix Figure A*n*. Prediction sheets for all saved test samples are published as [RSICC Model Run Reports](https://www.kaggle.com/datasets/kspsvlnsiddardha/rsicc-model-run-reports); sheets named predictions_phase7* contain Phase 6 outputs.

## A.1 Supplementary tables

Table: []{#tabA1}**Appendix Table A1.** Full LEVIR-CC test metrics (n = 1,929).

| Model | Loss | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | Cosine |
|---|---|---|---|---|---|---|---|---|
| Phase 1 | 0.9690 | 0.4720 | 0.3855 | 0.3393 | 0.3027 | 0.5966 | 0.6214 | 0.6477 |
| Phase 2 | 0.9345 | 0.5168 | 0.4237 | 0.3712 | 0.3284 | 0.6304 | 0.6545 | 0.7061 |
| Final | 0.8982 | 0.5311 | 0.4355 | 0.3810 | 0.3365 | 0.6392 | 0.6604 | 0.7123 |
| Phase 6 | 0.9225 | 0.5214 | 0.4297 | 0.3777 | 0.3358 | 0.6357 | 0.6624 | 0.7149 |
| Phase 8, stage 1a (epoch 7) | — | 0.5092 | 0.4167 | 0.3658 | 0.3250 | 0.6155 | 0.6482 | 0.7007 |
| Phase 8, stage 1b | — | 0 | 0 | 0 | 0 | 0 | 0 | 0.0104 |

Table: []{#tabA2}**Appendix Table A2.** Training configuration.

| Item | Phase 1 | Phase 2 | Final | Phase 6 | Phase 8 (1a) | Phase 8 (1b) |
|---|---|---|---|---|---|---|
| Input size (px) | 256 | 224 | 224 | 4 × 224 tiles | 224 | 224 |
| Batch size | 16 | 16 | 16 | 16 | 8 | 8 |
| Epochs | ≥ 10 | 10 | 15 | 15 | 15 | 10 |
| Learning rate | 1×10⁻⁴ | 1×10⁻⁴ | 1×10⁻⁴ | 1×10⁻⁴ | 1×10⁻⁴ | 5×10⁻⁵ |
| Weight decay | 1×10⁻⁵ | 1×10⁻⁵ | 1×10⁻⁵ | 1×10⁻⁵ | 1×10⁻⁵ | 1×10⁻⁵ |
| Vocabulary | 733 | 733 | 1,300 | 1,300 | 733 | Qwen tokenizer |

Table: []{#tabA3}**Appendix Table A3.** Forgetting after SECOND-CC fine-tuning (LEVIR-CC test set).

| Model | Loss | BLEU-4 | METEOR | ROUGE-L | Cosine |
|---|---|---|---|---|---|
| Final, before | 0.898 | 0.337 | 0.639 | 0.660 | 0.712 |
| Final, after | 4.080 | 0.149 | 0.394 | 0.437 | 0.458 |
| Phase 6, before | 0.923 | 0.336 | 0.636 | 0.662 | 0.715 |
| Phase 6, after | 3.520 | 0.121 | 0.318 | 0.374 | 0.438 |

Table: []{#tabA4}**Appendix Table A4.** Prediction behaviour on the saved prediction sheets (300 samples each).

| Prediction sheet | Changed pairs called "no difference" | Unchanged pairs correct | Distinct captions |
|---|---|---|---|
| Phase 6, LEVIR-CC | 43 / 164 | 98 / 136 | 21 |
| Phase 8, LEVIR-CC | 31 / 164 | 96 / 136 | 32 |
| Final after fine-tuning, LEVIR-CC | 164 / 164 | 136 / 136 | 1 |
| Phase 6, SECOND-CC | 16 / 255 | 33 / 45 | 97 |
| Phase 8, SECOND-CC | 19 / 255 | 30 / 45 | 82 |

Table: []{#tabA5}**Appendix Table A5.** Accepted and rejected patch pairs by region.

| Region | Accepted | Rejected |
|---|---|---|
| Deccan | 3,902 | 2,738 |
| Central-east | 3,179 | 2,485 |
| North plains | 2,655 | 2,129 |
| West desert | 2,095 | 2,033 |
| North-east | 1,580 | 804 |
| Himalaya | 1,529 | 1,095 |
| South and islands | 670 | 994 |
| **Total** | **15,610** | **12,278** |

Table: []{#tabA6}**Appendix Table A6.** How often each model called a pair "change", by human label (n = 495).

| Human label | n | Qwen3-VL-32B | Qwen3-VL-235B |
|---|---|---|---|
| Seasonal | 214 | 200 (93%) | 67 (31%) |
| No change | 226 | 184 (81%) | 53 (23%) |
| Change | 55 | 49 (89%) | 25 (45%) |

Table: []{#tabA7}**Appendix Table A7.** Results on all labelled pairs (n = 962; human label where available, automatic label otherwise).

| Rule | Recall | Precision | False-change rate |
|---|---|---|---|
| Qwen3-VL-32B | 96% | 26% | 47% |
| Qwen3-VL-235B | 79% | 48% | 15% |
| Both agree | 78% | 51% | 13% |

Table: []{#tabA8}**Appendix Table A8.** Training and validation loss summary. In every run, validation loss reaches its minimum early while training loss keeps falling.

| Run | Best validation loss | Final validation loss | Final training loss |
|---|---|---|---|
| Phase 2 | 1.045 (epoch 6) | 1.054 (epoch 10) | 0.716 |
| Phase 8, stage 1a | 1.017 (epoch 7) | 1.136 (epoch 15) | 0.482 (token accuracy 84.9%) |
| Phase 8, stage 1b | 1.417 (epoch 4) | 1.598 (epoch 10) | 0.839 |
| Final, SECOND-CC fine-tuning | 1.522 (epoch 6) | 1.551 (epoch 14) | 1.275 |

Table: []{#tabA9}**Appendix Table A9.** Full SECOND-CC test metrics (n = 1,227).

| Model / condition | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | Cosine |
|---|---|---|---|---|---|---|---|
| Phase 8 (1a), zero-shot | 0.179 | 0.128 | 0.103 | 0.081 | 0.298 | 0.300 | 0.378 |
| Phase 6, fine-tuned | 0.354 | 0.236 | 0.170 | 0.123 | 0.392 | 0.416 | 0.530 |
| Final, fine-tuned | 0.363 | 0.245 | 0.179 | 0.130 | 0.413 | 0.433 | 0.550 |
| Phase 8 (1b), zero-shot | 0 | 0 | 0 | 0 | 0 | 0 | 0.022 |

## A.2 Supplementary figures

![[]{#figA1}**Appendix Figure A1.** Training and validation loss for four runs: Phase 2, Phase 8 stage 1a, Phase 8 stage 1b and the Final model fine-tuned on SECOND-CC. Dots mark the best validation epoch.](figures/appA1_training_curves.png){width=100%}

![[]{#figA2}**Appendix Figure A2.** The same LEVIR-CC test pairs captioned by three runs, with the reference caption.](figures/appA2_levir_samples.png){width=95%}

![[]{#figA3}**Appendix Figure A3.** SECOND-CC test predictions (Phase 6 run).](figures/appA3_secondcc_samples.png){width=90%}

![[]{#figA4}**Appendix Figure A4.** Whole-image (left) and patch-wise (right) inference on unlabelled high-resolution pairs.](figures/appA4_patchwise.png){width=100%}

![[]{#figA5}**Appendix Figure A5.** A pair from the pilot Indian Sentinel-2 collection with its absolute-difference map.](figures/appA5_pilot_sample.png){width=100%}

![[]{#figA6}**Appendix Figure A6.** An Indian pilot pair with the answers of two off-the-shelf captioners.](figures/appA6_offtheshelf.png){width=80%}

![[]{#figA7}**Appendix Figure A7.** A human-confirmed change with pixel-difference and CNN region boxes; the plain question finds the change, the boxed prompts miss it.](figures/appA7_region_boxes.png){width=80%}

![[]{#figA8}**Appendix Figure A8.** Rule-based change map on a pair labelled seasonal, with the caption written from it.](figures/appA8_change_map.png){width=100%}

![[]{#figA9}**Appendix Figure A9.** Change gate: precision–recall area under the curve (PR-AUC) per feature set, site-grouped cross-validation.](figures/appA9_change_gate.png){width=85%}

![[]{#figA10}**Appendix Figure A10.** SECOND-CC test scores: zero-shot (Phase 8 stage 1a) and after fine-tuning (Phase 6, Final).](figures/appA10_secondcc_domain.png){width=85%}

# Appendix B: Data, code and model links {#appB}

All resources produced in this study are listed below. The Kaggle items are private until public release; access can be requested from the team.

Table: []{#tabB1}**Appendix Table B1.** Code repository, branches and notebooks.

| Item | Contents | Link |
|---|---|---|
| Repository (main branch) | All phases, India dataset pipeline, labelling and review tools, evaluation scripts | [github.com/Sankalp0109/…-Change-Detection](https://github.com/Sankalp0109/Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection) |
| Phase 1 notebook | CNN baseline, training and LEVIR-CC evaluation | [phase1.ipynb](https://github.com/Sankalp0109/Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection/blob/main/phase1.ipynb) |
| Phase 2 notebook | RemoteCLIP difference model | [phase2.ipynb](https://github.com/Sankalp0109/Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection/blob/main/phase2.ipynb) |
| Final model notebooks | Cross-attention model; SECOND-CC fine-tuning | [phase_final.ipynb](https://github.com/Sankalp0109/Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection/blob/main/phase_final.ipynb) · [phase_final_output_finetuneS.ipynb](https://github.com/Sankalp0109/Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection/blob/main/phase_final_output_finetuneS.ipynb) |
| Phase 6 notebook | Tile-based model | [phase6.ipynb](https://github.com/Sankalp0109/Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection/blob/main/phase6.ipynb) |
| Phase 7 / 8 (branch 8.1) | CodeAug design, Phase 8 stages 1a/1b, NaN fix | [phase7.ipynb](https://github.com/Sankalp0109/Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection/blob/8.1/phase7.ipynb) · [phase8.ipynb](https://github.com/Sankalp0109/Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection/blob/8.1/phase8.ipynb) · [phase8_finetune.ipynb](https://github.com/Sankalp0109/Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection/blob/8.1/phase8_finetune.ipynb) |
| Modular pipeline (branch modular) | Shared model interface, metrics, training code | [tree/modular](https://github.com/Sankalp0109/Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection/tree/modular) |
| Pilot Indian collection (branch rogue) | 60-area pilot download used in Part B | [tree/rogue](https://github.com/Sankalp0109/Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection/tree/rogue) |
| India dataset pipeline | Site sampling, download, screening, labelling experiments, 1,000-pair validation | [india_s2_dataset](https://github.com/Sankalp0109/Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection/tree/main/india_s2_dataset) |
| Review tool | Two-step blind labelling and caption review | [india_s2_dataset/labeler](https://github.com/Sankalp0109/Generative-and-Caption-Based-Vision-Language-Models-for-Remote-Sensing-Change-Detection/tree/main/india_s2_dataset/labeler) |

Table: []{#tabB2}**Appendix Table B2.** Model checkpoints for the models in [Table 4](#tab4).

| Model | Checkpoint files | Links |
|---|---|---|
| Phase 1 (CNN baseline) | researcher1_baseline_best.pt, shared_vocab.pkl | [Hugging Face file](https://huggingface.co/Kspsvln/IS/blob/main/researcher1_baseline_best.pt) · [Kaggle Models, variation pytorch/baseline](https://www.kaggle.com/models/kspsvlnsiddardha/rsicc-change-captioning) |
| Phase 2 (RemoteCLIP difference) | researcher2_remoteclip_difference_best.pt, shared_vocab.pkl | [Hugging Face file](https://huggingface.co/Kspsvln/IS/blob/main/researcher2_remoteclip_difference_best.pt) · [Kaggle Models, variation pytorch/remoteclip-difference](https://www.kaggle.com/models/kspsvlnsiddardha/rsicc-change-captioning) |
| Shared vocabulary | shared_vocab.pkl | [Hugging Face file](https://huggingface.co/Kspsvln/IS/blob/main/shared_vocab.pkl) |
| Final, Phase 6, Phase 8 | Held by the team; not yet uploaded | [Reserved repository huggingface.co/Kspsvln/IS_phase2](https://huggingface.co/Kspsvln/IS_phase2) |
| Phase 7 (CodeAug) | No trained checkpoint (design only) | — |

Table: []{#tabB3}**Appendix Table B3.** Datasets and prediction sheets released with this study.

| Resource | Contents | Link |
|---|---|---|
| India Sentinel-2 Change Pairs | 15,610 quality-screened pairs, PNG + GeoTIFF, site-level splits | [kaggle.com/datasets/kspsvlnsiddardha/india-sentinel2-change-pairs](https://www.kaggle.com/datasets/kspsvlnsiddardha/india-sentinel2-change-pairs) |
| India Sentinel-2 Change Pairs, labelled | 1,002 labelled pairs (535 human-labelled), with model answers | [kaggle.com/datasets/kspsvlnsiddardha/india-sentinel2-change-labelled](https://www.kaggle.com/datasets/kspsvlnsiddardha/india-sentinel2-change-labelled) |
| RSICC Model Run Reports | Prediction sheets of the LEVIR-CC and SECOND-CC runs | [kaggle.com/datasets/kspsvlnsiddardha/rsicc-model-run-reports](https://www.kaggle.com/datasets/kspsvlnsiddardha/rsicc-model-run-reports) |

Table: []{#tabB4}**Appendix Table B4.** Third-party datasets, pretrained models and services used.

| Resource | Used for | Link |
|---|---|---|
| LEVIR-CC | Part A training and testing | [github.com/Chen-Yang-Liu/RSICC](https://github.com/Chen-Yang-Liu/RSICC) |
| SECOND-CC | Part A transfer | [arxiv.org/abs/2501.10075](https://arxiv.org/abs/2501.10075) |
| RemoteCLIP ViT-B/32 | Frozen encoder (Phases 2–8) | [huggingface.co/chendelong/RemoteCLIP](https://huggingface.co/chendelong/RemoteCLIP) |
| Qwen2-0.5B-Instruct | Phase 8 stage 1b decoder | [huggingface.co/Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) |
| Qwen2-VL-2B-Instruct | CodeAug decoder | [huggingface.co/Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| TEOChat | Part B captioner | [huggingface.co/jirvin16/TEOChat](https://huggingface.co/jirvin16/TEOChat) |
| Qwen2.5-VL | Part B captioner | [github.com/QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) |
| Qwen3-VL-8B-Instruct | Haze screen; labelling tests | [huggingface.co/Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Qwen3-VL-32B-Instruct | Labelling tests; validation | [huggingface.co/Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| Qwen3-VL-235B-A22B-Instruct | Labelling tests; validation | [huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct) |
| Gemma 3 27B | Model-size comparison | [huggingface.co/google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) |
| Sentence-BERT all-MiniLM-L6-v2 | Caption cosine similarity | [huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Change-Agent | Related method (discussion) | [github.com/Chen-Yang-Liu/Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent) |
| Sentinel-2 L2A, Collection 1 | Indian imagery | [registry.opendata.aws/sentinel-2-l2a-cogs](https://registry.opendata.aws/sentinel-2-l2a-cogs/) |
| Earth Search STAC API | Scene catalogue | [earth-search.aws.element84.com/v1](https://earth-search.aws.element84.com/v1) |
| geoBoundaries | State boundaries for site sampling | [geoboundaries.org](https://www.geoboundaries.org) |
| OpenRouter | Hosted model inference | [openrouter.ai](https://openrouter.ai) |
