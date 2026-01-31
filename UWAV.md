# UWAV: Uncertainty-weighted Weakly-supervised Audio-Visual Video Parsing

Yung-Hsuan Lai1,†, Janek Ebbers3, Yu-Chiang Frank Wang1,2, Franc¸ois Germain3,Michael Jeffrey Jones3, Moitreya Chatterjee3,‡,1 Graduate Institute of Communication Engineering, National Taiwan University 2 NVIDIA, Taiwan3 Mitsubishi Electric Research Labs (MERL)†r10942097@ntu.edu.tw ‡metro.smiles@gmail.com

# Abstract

Audio-Visual Video Parsing (AVVP) entails the challeng-ing task of localizing both uni-modal events (i.e., those oc-curring exclusively in either the visual or acoustic modal-ity of a video) and multi-modal events (i.e., those occur-ring in both modalities concurrently). Moreover, the pro-hibitive cost of annotating training data with the class la-bels of all these events, along with their start and endtimes, imposes constraints on the scalability of AVVP tech-niques unless they can be trained in a weakly-supervisedsetting, where only modality-agnostic, video-level labelsare available in the training data. To this end, recently pro-posed approaches seek to generate segment-level pseudo-labels to better guide model training. However, the ab-sence of inter-segment dependencies when generating thesepseudo-labels and the general bias towards predicting la-bels that are absent in a segment limit their performance.This work proposes a novel approach towards overcom-ing these weaknesses called Uncertainty-weighted Weakly-supervised Audio-visual Video Parsing (UWAV). Addition-ally, our innovative approach factors in the uncertainty as-sociated with these estimated pseudo-labels and incorpo-rates a feature mixup based training regularization for im-proved training. Empirical results show that UWAV outper-forms state-of-the-art methods for the AVVP task on mul-tiple metrics, across two different datasets, attesting to itseffectiveness and generalizability.

# 1. Introduction

Events that occur in the real world, often leave their imprinton the acoustic and visual modalities. Humans rely heavilyon the synergy between their senses of sight and hearing tointerpret such events. Audio-visual learning, which seeksto equip machines with a similar synergy, has emerged asone of the most important research areas within the multi-modal machine learning community. It aims to leverage

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/70b3c3f503bf5a134a8443d5ede6cd498086a722fbafc153581086b3d2a9027f.jpg)



Figure 1. A weakly-supervised AVVP task example. Events,considered in this task, might be unimodal or multimodal. Evenmultimodal events, may not be temporally aligned in the audioand visual modalities, e.g. the cello might only be visible in thefirst few seconds but might produce music, throughout the video.


both these senses (modalities) jointly, to enhance machineperception and understanding of real-world events. Variousaudio-visual learning tasks have been studied towards thisend, including audio-visual segmentation [22, 52], soundsource localization [27, 35], audio-visual event localiza-tion [32, 37], and audio-visual sound separation [3, 7, 46].However, many of these tasks assume that audio and visualstreams would always be temporally aligned. This assump-tion often fails in real-world scenarios, where the sonic andvisual imprints of events may not perfectly overlap. Forinstance, one might hear an emergency siren approachingfrom a distance before it appears in the field of view.

In this work, in order to better understand the eventsoccurring in a video, we explore the task of Audio-VisualVideo Parsing (AVVP) [38]. Its goal is to recognize and lo-calize all audio, visual, and audio-visual events occurring inthe video. See Figure 1 for an example of this task. The tasksetup is to perform this prediction for every one-second seg-ment of a video. This task poses two principal challenges,from a machine learning standpoint: (i) The audio and vi-

sual events, that occur, might not be temporally aligned,e.g. if an event becomes audible before its source entersthe camera field of view, or the sound source is not visi-ble at all, and (ii) due to the high costs of annotating videosegments with per-segment labels, only modality-agnostic,video-level labels are provided during training, i.e., theselabels specify which events occur in a video but lack detailsabout the segments or the modality in which they occur.

Prior works in the area can be grouped into two orthog-onal research directions. The first focuses on enhancingmodel architectures [26, 48, 54]. Despite advancements inthis direction, the absence of fine-grained labels to guidethe model during training continues to pose an impedimenttowards the generalizability of such models. As a result,recent approaches have focused on the second directionof research which aims at generating richer pseudo-labelsfor improved training, either at the video-level [8, 42] orsegment-level [10, 20, 29, 53]. In particular, Rachavarapuet al. [29] propose prototype-based pseudo-labeling (PPL),which seeks to train a pseudo-label generation module inconjunction with a core inference module. However, due tothe lack of sufficient training data, this method struggles togeneralize. On the other hand, VPLAN [53], VALOR [20],and LSLD [10] leverage large-scale pre-trained foundationmodels, such as CLIP [31] and CLAP [43], along withground-truth video-level labels to generate segment-levelpseudo-labels for each of the two modalities. Audio/Visualsegments (e.g. the audio corresponding to the segment inquestion and the visual frame at the center of the segment)are fed into CLAP/CLIP, one segment at a time, to gener-ate these pseudo-labels. Despite the significant improve-ment that these pseudo-label generation methods achieved,the correctness of the generated labels is still limited, con-strained primarily by the lack of understanding of inter-segment dynamics. For instance, if a crowd is cheering in asegment of the video, it is more likely that the crowd mightalso be clapping right before or after.

To address the oversight of inter-segment dependen-cies and other shortcomings in existing pseudo-labelgeneration methods, we introduce a novel, uncertainty-based, weakly-supervised, video parsing model calledUWAV (Uncertainty-weighted Weakly-supervised Audio-visual Video Parsing), capable of generating improvedsegment-level pseudo-labels for better training of the in-ference module. We resort to transformer modules [41] toequip UWAV with the ability to capture temporal relation-ships between segments and pre-train it on a large-scale,supervised audio-visual event localization dataset [14].Subsequently, this pre-trained model is used to generatesegment-level pseudo-labels for each modality on a target,small-scale dataset which only provides weak (i.e., video-level) supervision. Such a design permits a more holis-tic understanding of the video, resulting in more accurate

pseudo-labels. Additionally, UWAV factors in the uncer-tainty associated with these estimated pseudo-labels in itsoptimization. That uncertainty is the result of the shiftin the domain of the target dataset, insufficient model ca-pacity, etc. and is computed at training time as the con-fidence scores associated with these labels. To furtherenhance the model’s ability to learn in this small-scale,weakly-supervised data regime, we also employ a featuremixup strategy. This approach adds more regularizationconstraints by training on mixed segment features alongsideinterpolated pseudo-labels, which not only lessens the in-fluence of noise but also enriches the training data, therebyreducing overfitting. Moreover, UWAV addresses a criti-cal class imbalance issue in the pseudo-label enriched train-ing data, viz. most event classes in any given segment ofa video are absent/negative (i.e., they do not occur), whileonly a handful of them do. This creates a natural bias in thetraining set, making it difficult to learn the positive events.To counter this, we propose a class-frequency aware re-weighting strategy that lays greater emphasis on the accu-rate classification and localization of positive events. Byincorporating these components into its design, our pro-posed method (UWAV) outperforms competing state-of-the-art approaches across two different datasets, viz. Look,Listen, and Parse (LLP) [38] and the Audio-Visual EventLocalization (AVE) [37], on multiple evaluation metrics.In summary, our contributions are the following:

• We introduce a novel, weakly-supervised method calledUWAV, capable of synthesizing temporally coherentpseudo-labels for the AVVP task.

• To the best of our knowledge, ours is the first method forthe AVVP task, which factors in the uncertainty associ-ated with the estimated pseudo-labels while also regular-izing it with a feature mixup strategy.

• UWAV outperforms competing state-of-the-art ap-proaches for the AVVP task, across two different datasetson multiple metrics which attest to its generalizability.

# 2. Related Works

Audio-Visual Learning: Audio-visual learning hasemerged as an area of active research, aiming to developmodels that synergistically integrate information from bothaudio and visual modalities for improved perception andunderstanding. Towards this end, various audio-visualtasks have been explored by the community so far, such asaudio-visual segmentation [22, 25, 45, 52], sound sourcelocalization [17, 18, 27, 35], event localization [32, 37, 51],navigation [4–6, 24, 47, 49], generation [28, 33, 44],question answering [13, 21, 36, 50], and sound sourceseparation [2, 7, 39, 46]. In this work, we focus on thetask of audio-visual video parsing (AVVP) where thegoal is to temporally localize events occurring in a video.Unlike many other audio-visual learning tasks, AVVP

does not assume that events are always aligned acrossmodalities. Some events could be exclusively uni-modalwhile others may have an audio-visual signature, whichrequires complex reasoning.

Audio-Visual Video Parsing (AVVP): To address thechallenges of the AVVP task [20, 29, 38], Tian et al.[38] proposed a Hybrid Attention Network (HAN) and alearnable Multi-modal Multiple Instance Learning (MMIL)pooling module. The HAN model facilitates the exchangeof information within and across modalities using self-attention and cross-attention layers, while the MMIL pool-ing module aggregates segment-level event probabilitiesfrom both modalities to produce video-level probabilities.Building on this foundation, recent works advanced the fieldfrom the following two perspectives. The first group ofstudies [26, 48, 54] focuses on enhancing model architec-tures. In particular, Mo and Tian [26] proposed the Multi-modal Grouping Network (MGN) to explicitly group se-mantically similar features within each modality to improvethe reasoning process, while Yu et al. [48] proposed theMulti-modal Pyramid Attentional Network (MM-Pyramid)to capture events of varying durations by extracting featuresat multiple temporal scales. Our proposed method is orthog-onal to this line of research and can be integrated with anyof these backbones.

The second direction focuses on generating pseudo-labels for improved training, either at the video-level [8, 42]or the segment-level [10, 20, 29, 53]. VPLAN [53],VALOR [20], and LSLD [10] utilize pre-trained CLIP [31]and CLAP [43] along with ground-truth video-level labelsto predict pseudo-labels for each modality on a per-segmentbasis. In contrast, PPL [29] uses the HAN model itself togenerate pseudo-labels by constructing prototype featuresfor each class and assigning labels to each segment based onthe similarity between its feature and the prototype features.While these pseudo-label generation methods have substan-tially improved model performance on the AVVP task, theystill exhibit some limitations. For instance, to derive accu-rate pseudo-labels, PPL might require a large enough train-ing set to learn good prototype features, which might posechallenges when applied to smaller datasets. Our proposedmethod overcomes this problem. On the other hand, meth-ods that leverage CLIP and CLAP to generate pseudo-labelsoften ignore temporal relationships between segments orthe uncertainty associated with these labels. Our work alsoseeks to plug this void.

# 3. Preliminaries

Problem Formulation: The AVVP task [38] aims to lo-calize all visible and/or audible events in each one-secondsegment of a video. Specifically, an audible video is splitinto $T$ one-second segments, denoted as $\{ V _ { t } , A _ { t } \} _ { t = 1 } ^ { T }$ . Each

segment is annotated with a pair of ground-truth labels$y _ { t } ^ { - } \in \{ 0 , 1 \} ^ { C } , y _ { t } ^ { a } \in \{ 0 , 1 \} ^ { \bar { C } }$ , where $y _ { t } ^ { v }$ denotes visualevents, $y _ { t } ^ { a }$ denotes audio events, and $C$ denotes the totalnumber of events in the pre-defined event set of the data.However, owing to the weakly-supervised nature of the tasksetup $( y _ { t } ^ { v } , y _ { t } ^ { a } )$ are unavailable during training. Instead, onlythe modality-agnostic, video-level labels $y \in \{ 0 , 1 \} ^ { C }$ areavailable, where 1 indicates the presence of an event at anytime (either in the audio or visual stream or both) while 0indicates an event’s absence in the video.

Pseudo-Label Based AVVP Framework: The HybridAttention Network (HAN) [38] is a commonly used modelfor the AVVP task. The model works by first utilizingpre-trained visual and audio backbones to extract featuresfrom the visual and audio segments respectively, which arethen projected to two $d$ -dimensional feature spaces. The re-sulting visual segment-level features are denoted by ${ \boldsymbol { F } } ^ { v } =$$\{ f _ { t } ^ { v } \} _ { t = 1 } ^ { \bar { T } } \in \mathbb { R } ^ { T \times d }$ , while the audio segment-level featuresare denoted by $F ^ { a } = \{ f _ { t } ^ { a } \} _ { t = 1 } ^ { T } \in \mathbb { R } ^ { T \times d }$ . These features areprovided as input to the HAN model. In the model, informa-tion across segments within a modality and across modali-ties is exchanged through self-attention and cross-attentionlayers, as shown below:

$$
\tilde {f} _ {t} ^ {v} = f _ {t} ^ {v} + \underbrace {\operatorname {A t t n} \left(f _ {t} ^ {v} , F ^ {v} , F ^ {v}\right)} _ {\text {S e l f - A t t e n t i o n}} + \underbrace {\operatorname {A t t n} \left(f _ {t} ^ {v} , F ^ {a} , F ^ {a}\right)} _ {\text {C r o s s - A t t e n t i o n}}, \tag {1}
$$

$$
\tilde {f} _ {t} ^ {a} = f _ {t} ^ {a} + \underbrace {\operatorname {A t t n} \left(f _ {t} ^ {a} , F ^ {a} , F ^ {a}\right)} _ {\text {S e l f - A t t e n t i o n}} + \underbrace {\operatorname {A t t n} \left(f _ {t} ^ {a} , F ^ {v} , F ^ {v}\right)} _ {\text {C r o s s - A t t e n t i o n}}, \tag {2}
$$

where $\mathsf { A t t n } ( Q , K , V )$ denotes the standard multi-head at-tention mechanism [41]. Finally a classifier, shared acrossboth modalities, transforms the visual segment-level fea-tures $\tilde { F } ^ { v } \equiv \{ \tilde { f } _ { t } ^ { v } \} _ { t = 1 } ^ { T } \in \mathbb { R } ^ { T \times d }$ (resp. audio segment-levelfeatures its $\{ z _ { t } ^ { v } \} _ { t = 1 } ^ { T } ~ \in ~ \mathbb { R } ^ { T \times C }$ $\tilde { F } ^ { a } ~ = ~ \{ \tilde { f } _ { t } ^ { a } \} _ { t = 1 } ^ { T } )$ =1(resp. audio segment-level logits into visual segment-level log-$\{ z _ { t } ^ { a } \} _ { t = 1 } ^ { T } )$ . Segment-level probabilities $\{ \bar { p _ { t } ^ { v } } \} _ { t = 1 } ^ { T } , \{ p _ { t } ^ { a } \} _ { t = 1 } ^ { T } \in$$\mathbb { R } ^ { T \times C }$ are then obtained by applying the sigmoid functionon $\{ z _ { t } ^ { v } \} _ { t = 1 } ^ { T }$ and $\{ z _ { t } ^ { a } \} _ { t = 1 } ^ { T }$ .

Since, only video-level labels $y$ are available duringtraining, Tian et al. [38] introduce an attentive MMIL pool-ing module to learn to predict video-level probabilities $p \in$$\mathbb { R } ^ { \bar { C } }$ :

$$
W _ {\text {m o d a l}} ^ {v, a} = \operatorname {S o f t m a x} _ {\text {m o d a l}} \left(\mathrm {F C} _ {\text {m o d a l}} \left(\tilde {F} ^ {v, a}\right)\right), \tag {3}
$$

$$
W _ {\text {t i m e}} ^ {v, a} = \operatorname {S o f t m a x} _ {\text {t i m e}} \left(\mathrm {F C} _ {\text {t i m e}} \left(\tilde {F} ^ {v, a}\right)\right), \tag {4}
$$

where $\mathrm { F C } _ { m o d a l }$ and $\mathrm { F C } _ { t i m e }$ are two learnable fully-connected layers, $\tilde { F } ^ { v , a } = \mathrm { S t a c k } ( \tilde { F } ^ { v } , \tilde { F } ^ { a } ) \in \mathbb { R } ^ { 2 \times T \times d }$ de-notes the stacked visual and audio features along the firstdimension, $\mathrm { S o f t m a x } _ { m o d a l } ( \cdot )$ denotes the softmax opera-tion along the modality dimension (i.e., across $v , a )$ , whileSoftmax $_ { t i m e } ( \cdot )$ denotes the softmax operation along the


Stage 1:Training the pseudo labelmodels on larger data


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/fbe996362ed0fc320475b17fd24f3cfef84052c420fcdcf37802f2074f6d60ae.jpg)



Stage 2: Uncertainty-weightedtraining on target dataset


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/6545a345c5c67561d929532f0b68e3ab3ff4022458c9cb0316abfc4bbdc6dc38.jpg)



Feature Mixup


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/d7d4b7bc1f378735aed12e70950b0df117f9d6bbefcfe8558a21292f779cd4f7.jpg)



Figure 2. UWAV framework: In stage 1, pseudo-label generation modules are equipped with the ability to capture temporal relationshipsbetween segments by pre-training on a large-scale, supervised, audio-visual event localization dataset. In stage 2, temporally coherent,uncertainty-weighted pseudo-labels, derived from the pre-trained pseudo-label generation module, are used to guide the learning of theinference model (HAN) aided by a class-balanced loss re-weighting and uncertainty-weighted feature mixup strategy. Note that we use thefeature mixup strategy in both modalities while we only show the breakdown of the mixup operation for the audio modality.


temporal dimension (i.e., across $1 , \ldots , T )$ . Video-levelprobabilities $p \in \mathbb { R } ^ { C }$ are then obtained via:

$$
p = \sum_ {m = \{v, a \}} \sum_ {t = 1} ^ {T} W _ {\text {m o d a l}} ^ {m, t} \odot W _ {\text {t i m e}} ^ {m, t} \odot p _ {t} ^ {m}, \tag {5}
$$

where $\odot$ denotes the element-wise product. The HANmodel is then optimized with the binary cross-entropy(BCE) loss between the estimated video-level probabilities$p$ and video-level labels y: $\mathcal { L } _ { v i d e o } = \mathbf { B C E } ( p , y )$ .

# 4. Proposed Approach

In this section, we detail our proposed approach (UWAV).At a high level, UWAV works by generating better segment-level pseudo-labels to improve the training of a multi-modaltransformer-based inference module, e.g. HAN. Moreover,UWAV factors in the uncertainty associated with thesepseudo-labels, addresses the imbalance in the training data,and introduces self-supervised regularization constraints,which all lead to better performance. Figure 2 shows anoverview of our proposed framework.

# 4.1. Temporally-Coherent Pseudo-Label Synthesis

One major issue that plagues prior works, based on pseudo-label generation [10, 20, 53], is that they do not capturethe temporal dependencies between neighboring segmentswhen generating the pseudo-labels. That is, the generatedpseudo-labels are not temporally coherent. To plug thisvoid, we propose to incorporate transformer modules [41]into the pseudo-label generation pipeline, which mapsCLIP/CLAP encodings of a segment’s visual frame/audioto pseudo-labels. Specifically, two separate transformersare introduced, one each for the visual/audio pseudo-labelsynthesis modules.

Pre-Training: Training transformers often requires suffi-ciently large training data while datasets commonly usedfor the weakly-supervised AVVP task are relatively small.To mitigate this challenge, we propose to first pre-trainthe transformer-equipped pseudo-label generation moduleon a large-scale, supervised, audio-visual event localizationdataset – the UnAV [14] dataset. Specifically, given an au-dible video of duration $T ^ { \prime }$ seconds from the pre-trainingsplit the video into , with corresponding $T ^ { \prime }$ one-second segmentsdio-visual event labels$\{ V _ { t } ^ { \prime } , A _ { t } ^ { \prime } \} _ { t = 1 } ^ { T ^ { \prime } }$$\bar { y } _ { t } ^ { a v ^ { \prime } } \in \bar { \{ 0 , 1 \} } ^ { C ^ { \prime } }$ , where 1 indicates the presence of an eventin both modalities and 0 its absence in at least one modal-ity, while $C ^ { \prime }$ denotes the total number of event classes in thepre-training dataset. Next, the video frame at the temporalcenter of the visual segment is transformed into visual fea-tures $G _ { 0 } ^ { v \prime } = \{ g _ { 0 , t } ^ { v } / \} _ { t = 1 } ^ { T ^ { \prime } } \in \mathbb { R } ^ { T ^ { \prime } \times d _ { 1 } }$ with CLIP’s [31] imageencoder. These features are then fed into the correspondingtransformer of the visual stream, consisting of $L$ encoderblocks, each block containing a self-attention layer, Layer-Norm [1] (LN), and a 2-layer feed-forward network (FFN):

$$
\tilde {G} _ {l} ^ {v ^ {\prime}} = \mathrm {L N} \left(G _ {l} ^ {v ^ {\prime}} + \operatorname {A t t n} \left(G _ {l} ^ {v ^ {\prime}}, G _ {l} ^ {v ^ {\prime}}, G _ {l} ^ {v ^ {\prime}}\right)\right), \tag {6}
$$

$$
G _ {l + 1} ^ {v ^ {\prime}} = \operatorname {L N} \left(\tilde {G} _ {l} ^ {v ^ {\prime}} + \operatorname {F F N} \left(\tilde {G} _ {l} ^ {v ^ {\prime}}\right)\right). \tag {7}
$$

Concurrently, we convert each event category label in thepre-training dataset into a textual event feature $e _ { c } ^ { C L I P ^ { \prime } } \in \mathbb { R } ^ { d _ { 1 } }$by filling in the pre-defined caption template: “A photo of<EVENT NAME>” with the corresponding event nameand passing it to CLIP’s text encoder. Equipped with thevisual segment-level features $G _ { L } ^ { v \prime } = \{ g _ { L , t } ^ { v } \} _ { t = 1 } ^ { \dot { T } ^ { \prime } } \in \mathbb { R } ^ { T ^ { \prime } \times d _ { 1 } }$ ∈ RT ′×d1and the textual event features ECLIP′ $E ^ { C L I P ^ { \prime } } = \{ e _ { c } ^ { C L I P ^ { \prime } } \} _ { c = 1 } ^ { C ^ { \prime } } \in $ {e CLIPc ′ } C ′c =1$\mathbb { R } ^ { C ^ { \prime } \times d _ { 1 } }$ , we derive visual segment-level logits $\hat { z } _ { t } ^ { v \prime } \in \mathbb { R } ^ { C ^ { \prime } }$ ∈ R C ′and probabilities $\hat { p } _ { t } ^ { v \prime }$ as follows:

$$
\hat {p} _ {t} ^ {v ^ {\prime}} = \operatorname {S i g m o i d} \left(\hat {z} _ {t} ^ {v ^ {\prime}}\right), \hat {z} _ {t} ^ {v ^ {\prime}} = E ^ {C L I P ^ {\prime}} \cdot g _ {L, t} ^ {v ^ {\prime} \top}. \tag {8}
$$

Similar operations are performed in the audio pseudo-label generation pipeline. The raw waveforms correspond-ing to the 1-second audio segments are transformed into au-dio features $G _ { 0 } ^ { a \prime } \in \mathbb { R } ^ { T ^ { \prime } \times d _ { 2 } }$ with CLAP’s [14] audio encoderand fed into the corresponding transformer consisting of $L$encoder blocks. Correspondingly, the textual event features$E ^ { C L A P ^ { \prime } } \in \mathbb { R } ^ { C ^ { \prime } \times d _ { 2 } }$ are generated with the caption template:“This is the sound of <EVENT NAME>” by passing itthrough CLAP’s text encoder. Audio segment-level logits$\hat { z } _ { t } ^ { a \prime } \in \mathbb { R } ^ { C ^ { \prime } }$ and probabilities $\hat { p } _ { t } ^ { a \prime }$ can then be derived in thesame manner: pˆat ′ = Sigmoid(ˆzat ′), zˆat ′ = ECLAP $\hat { p } _ { t } ^ { a \prime } = \mathrm { S i g m o i d } ( \hat { z } _ { t } ^ { a \prime } )$ $\hat { z } _ { t } ^ { a \prime } = E ^ { C L A P ^ { \prime } } \cdot g _ { t } ^ { a \prime } { } ^ { \top }$ ′ · g a ′⊤ .

Since the events occurring in the pre-training dataset(UnAV) are audio-visual, we multiply the segment-level vi-sual and audio event probabilities to enforce the predictedlabels to be multi-modal in nature: $\{ \hat { p } _ { t } ^ { a v \prime } \} _ { t = 1 } ^ { T ^ { \prime } } \in \mathbf { \hat { R } } ^ { T ^ { \prime } \times C ^ { \prime } }$ ′ }T ′t =1 ∈ R T ′ × C ′ .This network is then trained with the binary cross-entropy(BCE) loss:

$$
\mathcal {L} _ {\text {t e m p}} = \mathrm {B C E} \left(\hat {p} _ {t} ^ {a v ^ {\prime}}, y _ {t} ^ {a v ^ {\prime}}\right), \hat {p} _ {t} ^ {a v ^ {\prime}} = \hat {p} _ {t} ^ {v ^ {\prime}} \odot \hat {p} _ {t} ^ {a ^ {\prime}}. \tag {9}
$$

Pseudo-Label Generation on Target Dataset: With thepre-trained pseudo-label generation modules in place, weproceed to employ them for the pseudo-label generationprocess in the target dataset for the AVVP task. Specifically,the center frame of each of the visual segments $\{ V _ { t } \} _ { t = 1 } ^ { T }$ ofthe target dataset are passed into CLIP’s image encoder,whose output is passed into the pre-trained visual trans-former to generate segment features $G _ { L } ^ { v } \ = \ \{ g _ { L , t } ^ { v } \} _ { t = 1 } ^ { T } \ \in$$\mathbb { R } ^ { T \times d _ { 1 } }$ . At the same time, the caption template: “A photo of<EVENT NAME>” is used to obtain textual features cor-responding to each of the event classes in the target datasetfor the AVVP task: $E ^ { C L I P } \in \mathbb { R } ^ { C \times d _ { 1 } }$ . Segment-level visuallogits $\hat { z } _ { t } ^ { v } \in \mathbb { R } ^ { C }$ can be derived by computing their innerproduct. We also pre-define class-wise visual thresholds$\theta ^ { v } \in \mathbb { R } ^ { C }$ to transform segment-level visual logits into bi-nary pseudo-labels $\hat { y } _ { t } ^ { v } \in \mathbb { R } ^ { C }$ :

$$
\hat {y} _ {t} ^ {v} = \mathbb {1} _ {\left\{\hat {z} _ {t} ^ {v} > \theta^ {v} \right\}} \odot y, \hat {z} _ {t} ^ {v} = E ^ {C L I P} \cdot g _ {L, t} ^ {v} ^ {\top}, \tag {10}
$$

where $y$ denotes the ground-truth video-level labels, $\mathbb { 1 } _ { \{ \cdot \} }$ isthe indicator function which returns a value of 1 when thecondition is true otherwise 0, and $\odot$ denotes the element-wise product operation. The $\odot$ operation zeroes out thepredictions of event classes absent in the video-level label.

A similar pseudo-label generation process is employedon the acoustic side. Raw waveforms of audio segments arefirst fed into CLAP’s audio encoder and then into the pre-trained audio transformer. The event names of the classesin the target dataset for the AVVP task are filled in the cap-tion template: “This is the sound of <EVENT NAME>” togenerate textual event features: $E ^ { C L A P } \in \mathbb { R } ^ { C \times d _ { 1 } }$ . Segment-level audio logits $\hat { z } _ { t } ^ { a } \in \mathbb { R } ^ { C }$ and binary pseudo-labels $\hat { y } _ { t } ^ { a } \in$$\mathbb { R } ^ { C }$ are then derived using class-wise thresholds $\theta ^ { a } \in \mathbb { R } ^ { C }$ .

With binary segment-level pseudo-labels for both modal-ities $\hat { y } _ { t } ^ { v } , \hat { y } _ { t } ^ { a }$ and the predicted probabilities from the infer-

ence module (HAN) $\hat { p } _ { t } ^ { v } , \hat { p } _ { t } ^ { a }$ in place, the inference modulecan be trained using the binary cross-entropy loss as shown:

$$
\mathcal {L} _ {h a r d} = \mathrm {B C E} \left(p _ {t} ^ {v}, \hat {y} _ {t} ^ {v}\right) + \mathrm {B C E} \left(p _ {t} ^ {a}, \hat {y} _ {t} ^ {a}\right). \tag {11}
$$

# 4.2. Training with Pseudo-Label Uncertainty

While pseudo-labels do provide additional supervision forbetter training of the inference module, they could po-tentially be noisy, leading to occasionally incorrect train-ing signals. To ameliorate this problem, we propose anuncertainty-weighted pseudo-label training scheme to im-prove the robustness of the learning process. Instead of sim-ply training with the binary pseudo-labels $\hat { y } _ { t } ^ { v } , \hat { y } _ { t } ^ { a }$ , we lever-age the confidence of the pseudo-label estimation module(associated with the predicted pseudo-label) to weigh thetraining signal for the inference module. This confidencescore serves as a measure of the pseudo-label generationmodule’s uncertainty of its prediction. This may be repre-sented as:

$$
\hat {p} _ {t} ^ {v} = \operatorname {S i g m o i d} \left(\hat {z} _ {t} ^ {v} - \theta^ {v}\right) \odot y; \quad \hat {p} _ {t} ^ {a} = \operatorname {S i g m o i d} \left(\hat {z} _ {t} ^ {a} - \theta^ {a}\right) \odot y. \tag {12}
$$

In other words, considering the visual pseudo-label genera-tion pipeline as an example, the farther the logit $\hat { z } _ { t } ^ { v }$ is fromthe threshold $\theta ^ { v }$ , whether much lower or higher, the moreconfident the pseudo-label generation module is about thelabel it predicts (either approaching 0 or 1). Conversely,the closer the logit is to the threshold, the less the cer-tainty about the correctness of the pseudo-labels (probabil-ities closer to 0.5). An analogous explanation also holdsfor the audio pseudo-labels. With the uncertainty-weightedpseudo-labels in place, the inference module (HAN) canbe trained with the following uncertainty-weighted pseudo-label loss:

$$
\mathcal {L} _ {\text {s o f t}} = \mathrm {B C E} \left(p _ {t} ^ {v}, \hat {p} _ {t} ^ {v}\right) + \mathrm {B C E} \left(p _ {t} ^ {a}, \hat {p} _ {t} ^ {a}\right). \tag {13}
$$

# 4.3. Uncertainty-weighted Feature Mixup

Due to the lack of full supervision for the weakly-supervised AVVP task, we explore the efficacy of additionalregularization via self-supervision to help the models gener-alize better. Towards this end, prior pseudo-label generationapproaches [29, 42] often employ contrastive learning as atool to better train the inference module. However, due tothe inherent noise in the estimated pseudo-labels, positivesamples and negative samples may be mislabeled, decreas-ing the effectiveness of the self-supervisory training. Asan alternative, in this work, we explore the effectivenessof feature mixing, as a self-supervisory training signal foradditional regularization. In this setting, we mixup the esti-mated features of any two segments, additively, and train themodel to predict the union of the labels of the two segments.However, since the labels in our setting are noisy, the mixedfeature is assigned a label derived from a weighted sum of

the uncertainty-weighted pseudo-labels of each of the twosegment features. This is illustrated below:

$$
\bar {f} _ {t _ {i}, t _ {j}} ^ {v} = \lambda \tilde {f} _ {t _ {i}} ^ {v} + (1 - \lambda) \tilde {f} _ {t _ {j}} ^ {v}, \quad \bar {p} _ {t _ {i}, t _ {j}} ^ {v} = \lambda \hat {p} _ {t _ {i}} ^ {v} + (1 - \lambda) \hat {p} _ {t _ {j}} ^ {v} \tag {14}
$$

$$
\bar {f} _ {t _ {i}, t _ {j}} ^ {a} = \lambda \tilde {f} _ {t _ {i}} ^ {a} + (1 - \lambda) \tilde {f} _ {t _ {j}} ^ {a}, \quad \bar {p} _ {t _ {i}, t _ {j}} ^ {a} = \lambda \hat {p} _ {t _ {i}} ^ {a} + (1 - \lambda) \hat {p} _ {t _ {j}} ^ {a}, \tag {15}
$$

where $\lambda \sim \operatorname { B e t a } ( \alpha , \alpha )$ and $\alpha$ is a hyper-parameter control-ling the Beta distribution, and $t _ { i }$ and $t _ { j }$ indicate two segmentindices in a batch of video segments.

After mixing the uni-modal segment-level features, wepass them through the classifier of the inference mod-ule and apply the sigmoid function to the output, obtain-ing mixed segment-level event probabilities $p _ { t } ^ { m i x - v }$ and$p _ { t } ^ { m i x - a }$ . These are used to train the inference model withthe uncertainty-aware mixup loss, as shown below:

$$
\mathcal {L} _ {m i x} = \mathrm {B C E} \left(p _ {t} ^ {m i x - v}, \bar {p} _ {t} ^ {v}\right) + \mathrm {B C E} \left(p _ {t} ^ {m i x - a}, \bar {p} _ {t} ^ {a}\right). \tag {16}
$$

# 4.4. Class-balanced Loss Re-weighting

Besides the aforementioned challenges of the AVVP task,most of the events in the event set are absent in the pseudo-labels of any (segment of a) video (i.e., most event classesare negative events) and only a few events are present (i.e.,positive events are much fewer in number). As a result, themodel is dominated by the loss from the negative events.When trained without factoring in this bias, the classifiertends to overfit the negative labels and ignore the positiveones. To address this class imbalance issue, we introducea class-balanced loss re-weighting strategy to re-balancethe importance of the losses from the negative and posi-tive events for the uncertainty-weighted pseudo-label loss.Specifically, the loss from the positive events is multipliedby a weight proportional to the frequency of the segmentswith the negative events in the pseudo-labels, while the lossfrom the negative events is multiplied by a weight propor-tional to the frequency of the segments with the positiveevents in the pseudo-labels, as shown below:

$$
\begin{array}{l} \mathcal {L} _ {w - s o f t} = \sum_ {m \in \{v, a \}} w _ {p o s} ^ {m} \cdot y \cdot \mathrm {B C E} \left(p _ {t} ^ {m}, \hat {p} _ {t} ^ {m}\right) + \\ w _ {n e g} ^ {m} \cdot (1 - y) \cdot \mathrm {B C E} \left(p _ {t} ^ {m}, \hat {p} _ {t} ^ {m}\right), \tag {17} \\ \end{array}
$$

$$
w _ {p o s} ^ {m} = \frac {\sum_ {i = 1} ^ {N} \sum_ {t = 1} ^ {T} \sum_ {c = 1} ^ {C} \left(1 - \hat {y} _ {i , t , c} ^ {m}\right)}{N T C} \times W, \tag {18}
$$

$$
w _ {n e g} ^ {m} = \frac {\sum_ {i = 1} ^ {N} \sum_ {t = 1} ^ {T} \sum_ {c = 1} ^ {C} \hat {y} _ {i , t , c} ^ {m}}{N T C}, \tag {19}
$$

where $N$ denotes the number of videos in the training set,and $W$ is a hyper-parameter.

In summary, the inference module is trained on theAVVP task with the proposed class-balanced re-weighting,applied to the uncertainty-weighted classification loss, andthe uncertainty-weighted feature mixup loss, as shown be-low:

$$
\mathcal {L} _ {\text {t o t a l}} = \mathcal {L} _ {w - \text {s o f t}} + \mathcal {L} _ {\text {m i x}} + \mathcal {L} _ {\text {v i d e o}}. \tag {20}
$$

# 5. Experiments

We assess the performance of UWAV empirically acrosstwo challenging datasets and report its performance, com-paring it with existing state-of-the-art approaches bothquantitatively and qualitatively. Additionally, through mul-tiple ablation studies, we bring out the effectiveness of thedifferent elements of our proposed approach and the choicesof different hyper-parameters. For additional details, abla-tion studies, and more qualitative results, we refer the readerto our supplementary material.

# 5.1. Experimental Setup

Datasets: We evaluate all competing methods on theLook, Listen, and Parse (LLP) dataset [38], which is theprincipal benchmark dataset for the AVVP task. The datasetconsists of 11, 849 video clips sourced from YouTube. Eachclip is 10 seconds long and represents one or more of 25 di-verse event categories, such as human activities, animals,musical instruments, and vehicles. The dataset is split intotraining, validation, and testing sets, following the officialsplit [38]: 10, 000 videos for training, 649 videos for vali-dation, and 1, 200 videos for testing. While the training setof this dataset is only associated with video-level labels ofthe events, the validation and testing split is labeled withsegment-level event labels for evaluation purposes. Addi-tionally, to demonstrate the generalizability of our method,we conduct a similar study on the Audio Visual Event (AVE)recognition dataset [37]. The AVE dataset consists of 4, 143video clips crawled from YouTube, each 10 seconds long.It is split into 3, 339 videos for training, 402 for validation,and 402 for testing. It includes 29 event categories encom-passing human activities, animals, musical instruments, ve-hicles, and a “background” class (i.e., no event occurs). Un-like the LLP dataset, each video in the AVE dataset con-tains only one audio-visual event. Here too, the trainingdata is only provided with video-level labels while the vali-dation and test splits are annotated with ground-truth eventlabels for each one-second segment, which either is a spe-cific audio-visual event or “background”.

Metrics: For the LLP dataset, following the standardevaluation protocol [38], all models are evaluated usingmacro F1-scores calculated for the following event types:(i) audio-only (A), (ii) visual-only (V), and (iii) audio-visual(AV). Type@AV (Type) and Event@AV (Event) are two ad-ditional metrics that evaluate the overall performance of themodel, where Type@AV is the mean of the F1-scores forthe A, V, and AV events, while Event@AV is the F1-scoreof all events regardless of the modality in which they occur.Evaluations are conducted at both the segment-level and theevent-level. At the segment-level, the model’s predictionsare compared with the ground truth on a per-segment ba-sis. At the event-level, consecutive positive segments for


Table 1. Comparison with state-of-the-arts methods on the LLP dataset. Best performances are in bold, second-best in underlined.


<table><tr><td rowspan="2">Method</td><td colspan="5">Segment-level</td><td colspan="5">Event-level</td></tr><tr><td>A</td><td>V</td><td>AV</td><td>Type</td><td>Event</td><td>A</td><td>V</td><td>AV</td><td>Type</td><td>Event</td></tr><tr><td>HAN [38]</td><td>60.1</td><td>52.9</td><td>48.9</td><td>54.0</td><td>55.4</td><td>51.3</td><td>48.9</td><td>43.0</td><td>47.7</td><td>48.0</td></tr><tr><td>MA [42]</td><td>60.3</td><td>60.0</td><td>55.1</td><td>58.9</td><td>57.9</td><td>53.6</td><td>56.4</td><td>49.0</td><td>53.0</td><td>50.6</td></tr><tr><td>JoMoLD [8]</td><td>61.3</td><td>63.8</td><td>57.2</td><td>60.8</td><td>59.9</td><td>53.9</td><td>59.9</td><td>49.6</td><td>54.5</td><td>52.5</td></tr><tr><td>CMPAE [11]</td><td>64.2</td><td>66.4</td><td>59.2</td><td>63.3</td><td>62.8</td><td>56.6</td><td>63.7</td><td>51.8</td><td>57.4</td><td>55.7</td></tr><tr><td>PoiBin [30]</td><td>63.1</td><td>63.5</td><td>57.7</td><td>61.4</td><td>60.6</td><td>54.1</td><td>60.3</td><td>51.5</td><td>55.2</td><td>52.3</td></tr><tr><td>VPLAN [53]</td><td>60.5</td><td>64.8</td><td>58.3</td><td>61.2</td><td>59.4</td><td>51.4</td><td>61.5</td><td>51.2</td><td>54.7</td><td>50.8</td></tr><tr><td>VALOR [20]</td><td>61.8</td><td>65.9</td><td>58.4</td><td>62.0</td><td>61.5</td><td>55.4</td><td>62.6</td><td>52.2</td><td>56.7</td><td>54.2</td></tr><tr><td>LSLD [10]</td><td>62.7</td><td>67.1</td><td>59.4</td><td>63.1</td><td>62.2</td><td>55.7</td><td>64.3</td><td>52.6</td><td>57.6</td><td>55.2</td></tr><tr><td>PPL [29]</td><td>65.9</td><td>66.7</td><td>61.9</td><td>64.8</td><td>63.7</td><td>57.3</td><td>64.3</td><td>54.3</td><td>59.9</td><td>57.9</td></tr><tr><td>CoLeaf [34]</td><td>64.2</td><td>64.4</td><td>59.3</td><td>62.6</td><td>62.5</td><td>57.6</td><td>63.2</td><td>54.2</td><td>57.9</td><td>55.6</td></tr><tr><td>LEAP [54]</td><td>62.7</td><td>65.6</td><td>59.3</td><td>62.5</td><td>61.8</td><td>56.4</td><td>63.1</td><td>54.1</td><td>57.8</td><td>55.0</td></tr><tr><td>UWAV (Ours)</td><td>64.2</td><td>70.0</td><td>63.4</td><td>65.9</td><td>63.9</td><td>58.6</td><td>66.7</td><td>57.5</td><td>60.9</td><td>57.4</td></tr></table>

the same event are grouped together as a single event. TheF1-score is then computed using a mIoU threshold of 0.5.For the AVE dataset, we follow Tian et al. [37] and use ac-curacy as the evaluation metric. An event prediction of asegment is considered correct if it matches the ground-truthlabel for that segment.

Implementation Details: In line with prior work [38],each 10-second video in both datasets is split into 10 seg-ments of one second each, where each segment contains 8frames. For the LLP dataset, pre-trained ResNet-152 [15]and $\mathrm { R } ( 2 + 1 ) \mathrm { D } { - } 1 8 $ [40] are used to extract 2D and 3D vi-sual features, respectively. The pre-trained VGGish [16]network is used to extract features from the audio, sam-pled at $1 6 \mathrm { K H z }$ . For the AVE dataset however, akin toprior work [20], we extract visual features from pre-trainedCLIP and $\mathrm { R } ( 2 { + } 1 ) \mathrm { D }$ while CLAP is used to embed the au-dio stream. For both datasets, we set the number of en-coder blocks $L$ in the temporal-aware model to 5, $\alpha$ for theBeta distribution in the feature mixup strategy to 1.7, and$W$ in the class-balanced loss re-weighting step to 0.5. Thepseudo-label generation modules and the inference model(HAN) are trained with the AdamW optimizer [23]. For im-proved performance on the AVE dataset, we replace $\hat { p } _ { t _ { i } } ^ { a } , \hat { p } _ { t _ { j } } ^ { a }$in Eq. 15 with $\lceil \hat { p } _ { t _ { i } } ^ { a } \rceil , \lceil \hat { y } _ { t _ { j } } ^ { a } \rceil$ and make corresponding modifi-cations in the visual counterparts as well.

Baselines: We demonstrate the effectiveness of UWAVby comparing against an extensive set of baselines. Forthe LLP dataset, this includes video-level pseudo-labelgeneration methods (MA [42], JoMoLD [8]), segment-level pseudo-label generation methods (VALOR [20],LSLD [10], PPL [29]), and the recently released works(CoLeaf [34], LEAP [54]). On the other hand, for the AVEdataset, baseline approaches with publicly available imple-mentation, which use the state-of-the-art feature backbones(akin to ours), such as HAN [38] and VALOR [20] are used.

# 5.2. Results

Comparison with Previous Methods on LLP: Asshown in Table 1, UWAV surpasses previous methods,across almost all metrics. Notably, we achieve an F-scoreof 70.0 on the segment-level visual event, 65.9 on thesegment-level Type@AV, and 66.7 on the event-level vi-sual event metric. This corresponds to a gain of $1 . 1 \%$ onsegment-level Type@AV F-score and a $1 \%$ improvement onevent-level Type $@$ AV F-score, over our closest competitorPPL [29]. Of particular note, is the fact that our segment-level and event-level F-scores improve by more than $3 \%$ ,over PPL, for visual events. When compared to other re-cently published works, such as VALOR [20], CoLeaf [34]and LEAP [54], UWAV outperforms them by up to $3 \%$on both segment and event-level Type@AV F-score whilegains on visual events are up to $5 \%$ on segment-level F-score.

These observations are consistent with our qualitativecomparisons, as well. In the example on the left of Figure 3,our model successfully recognizes and temporally localizesthe lawn mower event visually, whereas VALOR [20] (a re-cent state-of-the-art approach with publicly available imple-mentation) misclassifies it as a chainsaw. Additionally, ourmodel also accurately localizes the intermittent sound of thelawn mower. In contrast, VALOR not only misclassifies thesound of the lawn mower as that of a chainsaw but also in-correctly predicts that someone is talking in the video. Inthe example on the right of Figure 3, our model does noterr in recognizing either the visual presence or the audiopresence of the telephone, while VALOR fails to accuratelypredict the events in either modality.

Comparison with Previous Methods on AVE: Todemonstrate the generalizability of our method, we evalu-ate UWAV on the AVE [37] dataset and compare its per-

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/6ac4fdef62b10eb65a19555c01d10f31c0544288897f51f4fd8c40a2c895dcdc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/508454ed353eaab6b55a763c38cf85f9ca9538feb39020cd3259683fb1cd4794.jpg)



Figure 3. Comparison between predictions by UWAV and competing AVVP methods on the LLP dataset. “GT”: ground truth.



Table 2. Model performances on the AVE dataset. CLIP,$\mathrm { R } ( 2 + 1 ) \mathrm { D } { - } 1 8 $ , and CLAP are used as feature backbones.


<table><tr><td>Method</td><td>HAN [38]</td><td>VALOR [20]</td><td>UWAV (Ours)</td></tr><tr><td>Acc.(%)</td><td>75.3</td><td>80.4</td><td>80.6</td></tr></table>


Table 3. Accuracy of the generated pseudo-labels on LLP.


<table><tr><td rowspan="2">Method</td><td colspan="5">Segment-level</td></tr><tr><td>A</td><td>V</td><td>AV</td><td>Type</td><td>Event</td></tr><tr><td>VALOR [20]</td><td>80.5</td><td>61.7</td><td>55.7</td><td>66.0</td><td>74.6</td></tr><tr><td>PPL [29]</td><td>61.7</td><td>61.8</td><td>57.5</td><td>60.6</td><td>59.4</td></tr><tr><td>UWAV (Ours)</td><td>78.4</td><td>74.5</td><td>65.5</td><td>72.8</td><td>78.4</td></tr></table>

formance with that of previous works. From Table 2, weobserve that with the same backbone features, UWAV sur-passes VALOR, our closest competitor, even on this small-scale dataset.

Accuracy of the Generated Pseudo-Labels: To evalu-ate the efficacy of our pseudo-label generation pipeline,we compare the accuracy of our generated pseudo-labelsagainst those obtained from competing methods (with pub-licly available implementation) [20, 29] on the test set ofthe LLP dataset. As shown in Table 3, our pre-trainedtemporally-dependent pseudo-label generation scheme gen-erates more accurate segment-level pseudo-labels than pre-vious methods, by up to $6 \%$ on the segment-level Type@AVF-score, attesting to the advantages of factoring in inter-segment temporal dependencies.

# 5.3. Ablation Study

To demonstrate the potency of the different elements of ourproposed method, UWAV, we conduct ablation studies. Inparticular, the proposed uncertainty-weighted pseudo-labelbased training, the uncertainty-weighted feature mixupscheme, and the class-balanced loss re-weighting schemesare ablated. As shown in Table 4, incorporating theuncertainty-weighted pseudo-label training step improves


Table 4. Ablation study of the proposed components in UWAV.“Binary” denotes training with binary pseudo-labels. “Soft” de-notes training with uncertainty-weighted pseudo-labels.


<table><tr><td rowspan="2">Binary</td><td rowspan="2">Soft</td><td rowspan="2">Re-weight</td><td rowspan="2">Mixup</td><td colspan="5">Segment-level</td></tr><tr><td>A</td><td>V</td><td>AV</td><td>Type</td><td>Event</td></tr><tr><td>✓</td><td></td><td></td><td></td><td>62.7</td><td>67.7</td><td>61.9</td><td>64.2</td><td>62.2</td></tr><tr><td></td><td>✓</td><td></td><td></td><td>63.0</td><td>68.3</td><td>61.8</td><td>64.4</td><td>62.8</td></tr><tr><td></td><td>✓</td><td>✓</td><td></td><td>63.6</td><td>69.5</td><td>63.0</td><td>65.4</td><td>63.1</td></tr><tr><td></td><td>✓</td><td></td><td>✓</td><td>63.9</td><td>69.0</td><td>62.8</td><td>65.2</td><td>63.4</td></tr><tr><td></td><td>✓</td><td>✓</td><td>✓</td><td>64.2</td><td>70.0</td><td>63.4</td><td>65.9</td><td>63.9</td></tr></table>

the segment-level Type@AV F-score by $2 \%$ , compared tousing binary pseudo-labels. This demonstrates the benefitof accounting for the uncertainty in the pseudo-label esti-mation module. Moreover, sans the class-balanced loss re-weighting strategy, the model’s performance is worse off by$1 \%$ on the Type@AV F-score, revealing the erroneous biasin the model’s prediction arising from a skew of the classdistribution. On the other hand, introducing the uncertainty-weighted feature mixup results in a gain of $0 . 8 \%$ on theType $@$ AV F-score, which underscores the importance ofthis self-supervised regularization.

# 6. Conclusions

In this work, we address the challenging task of weakly-supervised audio-visual video parsing (AVVP), whichpresents a two-fold challenge: (i) potential misalignmentbetween the events of the audio and visual streams, and (ii)the lack of fine-grained labels for each modality. We ob-serve that by considering the temporal relationship betweensegments, our proposed method (UWAV) is able to providemore reliable pseudo-labels for better training of the infer-ence module. In addition, by factoring in the uncertaintyassociated with these estimated pseudo-labels, regularizingthe training process with a feature mixup strategy, and cor-recting for class imbalance, UWAV achieves state-of-the-artresults on the LLP and AVE datasets.

# References



[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hin-ton. Layer normalization. arXiv preprint arXiv:1607.06450,2016. 4





[2] Moitreya Chatterjee, Jonathan Le Roux, Narendra Ahuja,and Anoop Cherian. Visual scene graphs for audio sourceseparation. In ICCV, 2021. 2





[3] Moitreya Chatterjee, Narendra Ahuja, and Anoop Cherian.Learning audio-visual dynamics using scene graphs for au-dio source separation. In NeurIPS, 2022. 1





[4] Changan Chen, Unnat Jain, Carl Schissler, Sebastia Vi-cenc Amengual Gari, Ziad Al-Halah, Vamsi Krishna Ithapu,Philip Robinson, and Kristen Grauman. Soundspaces:Audio-visual navigation in 3d environments. In ECCV, 2020.2





[5] Changan Chen, Ziad Al-Halah, and Kristen Grauman. Se-mantic audio-visual navigation. In CVPR, 2021.





[6] Changan Chen, Sagnik Majumder, Ziad Al-Halah, RuohanGao, Santhosh Kumar Ramakrishnan, and Kristen Grauman.Learning to set waypoints for audio-visual navigation. InICLR, 2021. 2





[7] Jiaben Chen, Renrui Zhang, Dongze Lian, Jiaqi Yang, ZiyaoZeng, and Jianbo Shi. iquery: Instruments as queries foraudio-visual sound separation. In CVPR, 2023. 1, 2





[8] Haoyue Cheng, Zhaoyang Liu, Hang Zhou, Chen Qian,Wayne Wu, and Limin Wang. Joint-modal label denoisingfor weakly-supervised audio-visual video parsing. In ECCV,2022. 2, 3, 7





[9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,and Li Fei-Fei. Imagenet: A large-scale hierarchical imagedatabase. In CVPR, 2009. 1





[10] Yingying Fan, Yu Wu, Yutian Lin, and Bo Du. Revisitweakly-supervised audio-visual video parsing from the lan-guage perspective. In NeurIPS, 2023. 2, 3, 4, 7





[11] Junyu Gao, Mengyuan Chen, and Changsheng Xu. Col-lecting cross-modal presence-absence evidence for weakly-supervised audio-visual event perception. In CVPR, 2023.





[12] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, ArenJansen, Wade Lawrence, R Channing Moore, Manoj Plakal,and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In ICASSP, 2017. 1





[13] Shijie Geng, Peng Gao, Moitreya Chatterjee, Chiori Hori,Jonathan Le Roux, Yongfeng Zhang, Hongsheng Li, andAnoop Cherian. Dynamic graph representation learning forvideo dialog via multi-modal shuffled transformers. In AAAI,2021. 2





[14] Tiantian Geng, Teng Wang, Jinming Duan, Runmin Cong,and Feng Zheng. Dense-localizing audio-visual events inuntrimmed videos: A large-scale benchmark and baseline.In CVPR, 2023. 2, 4, 5





[15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.Deep residual learning for image recognition. In CVPR,2016. 7, 1





[16] Shawn Hershey, Sourish Chaudhuri, Daniel PW Ellis, Jort FGemmeke, Aren Jansen, R Channing Moore, Manoj Plakal,





Devin Platt, Rif A Saurous, Bryan Seybold, et al. Cnn ar-chitectures for large-scale audio classification. In ICASSP,2017. 7, 1





[17] Xixi Hu, Ziyang Chen, and Andrew Owens. Mix and local-ize: Localizing sound sources in mixtures. In CVPR, 2022.2





[18] Chao Huang, Yapeng Tian, Anurag Kumar, and ChenliangXu. Egocentric audio-visual object localization. In CVPR,2023. 2





[19] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang,Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola,Tim Green, Trevor Back, Paul Natsev, et al. The kinetics hu-man action video dataset. arXiv preprint arXiv:1705.06950,2017. 1





[20] Yung-Hsuan Lai, Yen-Chun Chen, and Frank Wang.Modality-independent teachers meet weakly-supervisedaudio-visual event parser. In NeurIPS, 2023. 2, 3, 4, 7, 8,





[21] Guangyao Li, Yake Wei, Yapeng Tian, Chenliang Xu, Ji-Rong Wen, and Di Hu. Learning to answer questions in dy-namic audio-visual scenarios. In CVPR, 2022. 2





[22] Jinxiang Liu, Yikun Liu, Fei Zhang, Chen Ju, Ya Zhang, andYanfeng Wang. Audio-visual segmentation via unlabeledframe exploitation. In CVPR, 2024. 1, 2





[23] Ilya Loshchilov and Frank Hutter. Decoupled weight decayregularization. In ICLR, 2019. 7, 1





[24] Sagnik Majumder, Ziad Al-Halah, and Kristen Grauman.Move2hear: Active audio-visual source separation. In ICCV,2021. 2





[25] Yuxin Mao, Jing Zhang, Mochu Xiang, Yiran Zhong, andYuchao Dai. Multimodal variational auto-encoder basedaudio-visual segmentation. In ICCV, 2023. 2





[26] Shentong Mo and Yapeng Tian. Multi-modal grouping net-work for weakly-supervised audio-visual video parsing. InNeurIPS, 2022. 2, 3





[27] Shentong Mo and Yapeng Tian. Audio-visual grouping net-work for sound localization from mixtures. In CVPR, 2023.1, 2





[28] Kranti Kumar Parida, Siddharth Srivastava, and GauravSharma. Beyond mono to binaural: Generating binaural au-dio from mono audio with depth and cross modal attention.In WACV, 2022. 2





[29] Kranthi Kumar Rachavarapu, Kalyan Ramakrishnan,et al. Weakly-supervised audio-visual video parsing withprototype-based pseudo-labeling. In CVPR, 2024. 2, 3, 5, 7,8





[30] Kranthi Kumar Rachavarapu et al. Boosting positive seg-ments for weakly-supervised audio-visual video parsing. InICCV, 2023. 7





[31] Alec Radford, Jong Wook Kim, Chris Hallacy, AdityaRamesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn-ing transferable visual models from natural language super-vision. In ICML, 2021. 2, 3, 4, 1





[32] Varshanth Rao, Md Ibrahim Khalil, Haoda Li, Peng Dai, andJuwei Lu. Dual perspective network for audio-visual eventlocalization. In ECCV, 2022. 1, 2





[33] Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu,Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo.Mm-diffusion: Learning multi-modal diffusion models forjoint audio and video generation. In CVPR, 2023. 2





[34] Faegheh Sardari, Armin Mustafa, Philip JB Jackson, andAdrian Hilton. Coleaf: A contrastive-collaborative learningframework for weakly supervised audio-visual video pars-ing. In ECCV, 2024. 7, 1





[35] Arda Senocak, Hyeonggon Ryu, Junsik Kim, Tae-Hyun Oh,Hanspeter Pfister, and Joon Son Chung. Sound source local-ization is all about cross-modal alignment. In ICCV, 2023.1, 2





[36] Ankit Shah, Shijie Geng, Peng Gao, Anoop Cherian, TakaakiHori, Tim K Marks, Jonathan Le Roux, and Chiori Hori.Audio-visual scene-aware dialog and reasoning using audio-visual transformers with joint student-teacher learning. InICASSP, 2022. 2





[37] Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chen-liang Xu. Audio-visual event localization in unconstrainedvideos. In ECCV, 2018. 1, 2, 6, 7, 3





[38] Yapeng Tian, Dingzeyu Li, and Chenliang Xu. Unified mul-tisensory perception: Weakly-supervised audio-visual videoparsing. In ECCV, 2020. 1, 2, 3, 6, 7, 8





[39] Yapeng Tian, Di Hu, and Chenliang Xu. Cyclic co-learningof sounding object visual grounding and sound separation.In CVPR, 2021. 2





[40] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, YannLeCun, and Manohar Paluri. A closer look at spatiotemporalconvolutions for action recognition. In CVPR, 2018. 7, 1





[41] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-reit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and IlliaPolosukhin. Attention is all you need. In NeurIPS, 2017. 2,3, 4





[42] Yu Wu and Yi Yang. Exploring heterogeneous clues forweakly-supervised audio-visual video parsing. In CVPR,2021. 2, 3, 5, 7





[43] Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, TaylorBerg-Kirkpatrick, and Shlomo Dubnov. Large-scale con-trastive language-audio pretraining with feature fusion andkeyword-to-caption augmentation. In ICASSP, 2023. 2, 3, 1





[44] Yazhou Xing, Yingqing He, Zeyue Tian, Xintao Wang, andQifeng Chen. Seeing and hearing: Open-domain visual-audio generation with diffusion latent aligners. In CVPR,2024. 2





[45] Qi Yang, Xing Nie, Tong Li, Pengfei Gao, Ying Guo, ChengZhen, Pengfei Yan, and Shiming Xiang. Cooperation doesmatter: Exploring multi-order bilateral relations for audio-visual segmentation. In CVPR, 2024. 2





[46] Yuxin Ye, Wenming Yang, and Yapeng Tian. Lavss:Location-guided audio-visual spatial audio separation. InWACV, 2024. 1, 2





[47] Abdelrahman Younes, Daniel Honerkamp, TimWelschehold, and Abhinav Valada. Catch me if youhear me: Audio-visual navigation in complex unmappedenvironments with moving sounds. IEEE Robotics andAutomation Letters, 2023. 2





[48] Jiashuo Yu, Ying Cheng, Rui-Wei Zhao, Rui Feng, and Yue-jie Zhang. Mm-pyramid: Multimodal pyramid attentionalnetwork for audio-visual event localization and video pars-ing. In ACM MM, 2022. 2, 3





[49] Yinfeng Yu, Wenbing Huang, Fuchun Sun, Changan Chen,Yikai Wang, and Xiaohong Liu. Sound adversarial audio-visual navigation. In ICLR, 2022. 2





[50] Heeseung Yun, Youngjae Yu, Wonsuk Yang, Kangil Lee, andGunhee Kim. Pano-avqa: Grounded audio-visual questionanswering on 360deg videos. In ICCV, 2021. 2





[51] Jinxing Zhou, Liang Zheng, Yiran Zhong, Shijie Hao, andMeng Wang. Positive sample propagation along the audio-visual event line. In CVPR, 2021. 2





[52] Jinxing Zhou, Jianyuan Wang, Jiayi Zhang, Weixuan Sun,Jing Zhang, Stan Birchfield, Dan Guo, Lingpeng Kong,Meng Wang, and Yiran Zhong. Audio–visual segmentation.In ECCV, 2022. 1, 2





[53] Jinxing Zhou, Dan Guo, Yiran Zhong, and Meng Wang. Im-proving audio-visual video parsing with pseudo visual labels.arXiv preprint arXiv:2303.02344, 2023. 2, 3, 4, 7





[54] Jinxing Zhou, Dan Guo, Yuxin Mao, Yiran Zhong, Xiao-jun Chang, and Meng Wang. Label-anticipated event dis-entanglement for audio-visual video parsing. arXiv preprintarXiv:2407.08126, 2024. 2, 3, 7



# UWAV: Uncertainty-weighted Weakly-supervised Audio-Visual Video ParsingSupplementary Material

We begin this supplementary document by expoundingon the limitations of our proposed UWAV method. In thesection that follows, we elaborate on the implementationdetails, compute environment used for implementation, andthe training and inference times of our proposed method.Then, we quantitatively compare with VALOR on the LLPdatset using better backbone features. In Section 12, we putforward studies showcasing the sensitivity of our methodto the choice of the hyper-parameters $\alpha , W$ , followed byablation studies about the various design choices of ourmodel. Finally, we end this document by providing somequalitative visualizations of the predictions obtained by ourmethod versus competing baselines on both the LLP and theAVE datasets.

The following summarizes the supplementary materialswe provide:

• Limitations.

• Implementation details of UWAV.

• Details of our compute environment.

• Compute time analysis.

• Quantitative comparison using better backbone features.

• Studies on the sensitivity of UWAV to the choice of $\alpha , W$ .

• The Scalability of UWAV.

• Ablation studies on the different design choices.

• Qualitative results of UWAV versus competing methodsfor the AVVP task.

# 7. Limitations

Although UWAV achieves state-of-the-art results on theAVVP task, compared to competing methods, it requiresadditional training data to pre-train the pseudo-label gen-eration module (on which we train for about 80 epochs).

# 8. Implementation Details of UWAV

To assess the effectiveness of our method, in line with priorwork [38], each 10-second video in both the LLP [38] andAVE [37] datasets is split into 10 segments of one sec-ond each, where each segment contains 8 frames. Thevisual feature backbone for the LLP dataset is based onthe ResNet-152 [15] network (pre-trained on the ImageNetdataset [9]) for extracting 2D-appearance features, and the$\mathrm { R } ( 2 { + } 1 ) \mathrm { D }$ [40] network (pre-trained on the Kinetics-400dataset [19]) for extracting features that capture the visualdynamics, respectively. The VGGish [16] network, pre-trained on the AudioSet dataset [12], is used to extract fea-tures from the audio, sampled at 16KHz. For the AVEdataset however, akin to prior work [20], we extract visualfeatures from pre-trained CLIP [31] and $\mathrm { R } ( 2 { + } 1 ) \mathrm { D }$ , while


Table A5. Compute time analysis on the LLP dataset. “Infer-ence Time” denotes the time to evaluate all testing data.


<table><tr><td>Method</td><td>Training Time per Epoch</td><td>Inference Time</td></tr><tr><td>CoLeaf [34]</td><td>25 sec</td><td>24 sec</td></tr><tr><td>UWAV (Ours)</td><td>24 sec</td><td>20 sec</td></tr></table>

CLAP [43] is used to embed the audio stream. For bothdatasets, we set the number of encoder blocks $L$ of the trans-formers in the pseudo-label generation module to 5, $\alpha$ forthe Beta distribution in the feature mixup strategy to 1.7,and W in the class-balanced loss re-weighting step to 0.5.Both the pseudo-label generation modules and the inferencemodules are trained with the AdamW optimizer [23]. Totrain the model, we employ a learning rate scheduling strat-egy that warms up the learning rate for the first 10 epochsto its peak of 1e−4 and then decays according to a cosineannealing schedule, to the minimum, which is set to 1e−5for the pseudo-label generation models and 5e−6 for theinference model. We clip the gradient norm at 1.0 duringtraining. For the LLP dataset, the training batch size is setto 64 and the total number of training epochs to 80 for bothmodels, while the same is set to 16 and 80 for the AVEdataset.

# 9. Details of Compute Environment

Our model is trained on a desktop computer with an IntelCore i7 CPU, with 32GB RAM, and a single NVIDIA RTX3090 GPU.

# 10. Analysis of Compute Time

For a more holistic understanding of the performance of ourmethod, we compare its training and inference times withthe most recently published approach for the AVVP task,viz. CoLeaf [34] on the LLP dataset [38]. The results ofthis study are shown in Table A5. We see that our method’sruntime performances are comparable with those of com-peting approaches, with notable inference time gains overthe CoLeaf method [34].

# 11. Quantitative Comparison Using BetterBackbone Features

We also quantitatively compare our proposed approach withVALOR on the LLP dataset using better backbone features,i.e. CLIP and CLAP as visual and audio feature backbones.As shown in Table A6, UWAV outperforms VALOR with

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/502e2f7251a22f6a588a1fbf47577c0b8d760c1e259124b6e76c176dccd70261.jpg)



Figure A4. Sensitivity of $\alpha$ in the uncertainty-weighted featuremixup on the LLP dataset.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/1dd4024001c4dd34bab2d870c71c15108a245b140614918f26533bdb33611d13.jpg)



Figure A5. Sensitivity of $W$ in the class-balanced re-weightingon the LLP dataset.



Table A6. Comparison with VALOR on the LLP dataset. † denotes using CLIP and CLAP features as input to the HAN model.


<table><tr><td rowspan="2">Method</td><td colspan="5">Segment-level</td><td colspan="5">Event-level</td></tr><tr><td>A</td><td>V</td><td>AV</td><td>Type</td><td>Event</td><td>A</td><td>V</td><td>AV</td><td>Type</td><td>Event</td></tr><tr><td>VALOR†[17]</td><td>68.1</td><td>68.4</td><td>61.9</td><td>66.2</td><td>66.8</td><td>61.2</td><td>64.7</td><td>55.5</td><td>60.4</td><td>59.0</td></tr><tr><td>UWAV†(Ours)</td><td>68.9</td><td>72.3</td><td>65.6</td><td>68.9</td><td>68.3</td><td>63.5</td><td>68.7</td><td>59.6</td><td>63.9</td><td>62.4</td></tr></table>

2.7 F-score improvement in segment-level Type@AV and3.5 F-score improvement in event-level Type@AV.

# 12. Sensitivity to the Choice of $\alpha$ and $W$

To gain a better understanding of the effect of the choiceof hyper-parameters on our model’s performance, we eval-uate the sensitivity of our model to the choice $\alpha$ in theuncertainty-weighted feature mixup and $W$ in the class-balanced loss re-weighting strategy. When $\alpha$ is adjusted,class-balanced loss re-weighting is not applied. As shownin Figure A4, for the LLP dataset, when $\alpha$ increases from0.1 to 2.0, segment-level Type@AV F-score first decreasesto 64.5, then rises to a peak of 65.2 at $\alpha = 1 . 7$ , and sub-sequently declines back to 64.5. On the other hand, Fig-ure A5 illustrates the effect of varying W on the segment-level Type@AV F-score. The F-score reaches its maximumvalue of 65.3 when $W = 0 . 5$ and decreases as W becomeslarger. When W is adjusted, the uncertainty-weighted fea-ture mixup is not applied. These observations point towardsthe robustness of our model to the precise choice of thesehyper-parameters. We observe similar trends for the AVEdataset as well. Hence, for best results, we select $\alpha = 1 . 7$and $W = 0 . 5$ in all our experiments for both datasets.

# 13. The Scalability of UWAV

To evaluate the scalability of UWAV, we train the inferencemodel (HAN) with less training data (Table A7b) as well as


Table A7. The scalability of UWAV.



(a) Training with different amounts of data.


<table><tr><td rowspan="2">Training Data Ratio</td><td colspan="5">Segment-level</td></tr><tr><td>A</td><td>V</td><td>AV</td><td>Type</td><td>Event</td></tr><tr><td>100%</td><td>64.2</td><td>70.0</td><td>63.4</td><td>65.9</td><td>63.9</td></tr><tr><td>80%</td><td>63.4</td><td>69.2</td><td>62.5</td><td>65.0</td><td>63.0</td></tr><tr><td>60%</td><td>63.4</td><td>68.6</td><td>62.4</td><td>64.8</td><td>62.8</td></tr></table>


(b) Training with different number of classes.


<table><tr><td></td><td colspan="3">Segment-level Type F-score</td></tr><tr><td>Number of Classes</td><td>25 (all events)</td><td>20</td><td>15</td></tr><tr><td>VALOR [17]</td><td>62.0</td><td>65.9</td><td>66.6</td></tr><tr><td>UWAV(Ours)</td><td>65.9</td><td>71.4</td><td>68.4</td></tr></table>

fewer event classes (Table A7a) on the LLP dataset by re-moving the training videos or event classes randomly. Evenwith only $6 0 \%$ of the training data, UWAV exhibits com-petitive performance. Moreover, UWAV shows a consistentperformance lead against VALOR [17], irrespective of thenumber of event classes, with no change in training strategyor the core model structure.

# 14. Ablation Studies

Ablation Study on All Metrics: In Table A8, we reportthe ablation study on all metrics for a more complete un-


Table A8. Ablation study reported on all metrics. “Binary” denotes training with binary pseudo-labels. “Soft” denotes training withuncertainty-weighted pseudo-labels.


<table><tr><td rowspan="2">Binary Soft</td><td rowspan="2">Re-weight Mixup</td><td colspan="5">Segment-level</td><td colspan="5">Event-level</td></tr><tr><td>A</td><td>V</td><td>AV</td><td>Type</td><td>Event</td><td>A</td><td>V</td><td>AV</td><td>Type</td><td>Event</td></tr><tr><td>✓</td><td></td><td>62.7</td><td>67.7</td><td>61.9</td><td>64.2</td><td>62.2</td><td>56.9</td><td>64.9</td><td>56.6</td><td>59.5</td><td>55.8</td></tr><tr><td>✓</td><td></td><td>63.0</td><td>68.3</td><td>61.8</td><td>64.4</td><td>62.8</td><td>56.9</td><td>65.2</td><td>55.9</td><td>59.3</td><td>56.1</td></tr><tr><td>✓</td><td>✓</td><td>63.6</td><td>69.5</td><td>63.0</td><td>65.4</td><td>63.1</td><td>57.9</td><td>66.4</td><td>57.0</td><td>60.4</td><td>56.9</td></tr><tr><td>✓</td><td></td><td>63.9</td><td>69.0</td><td>62.8</td><td>65.2</td><td>63.4</td><td>57.7</td><td>65.6</td><td>56.3</td><td>59.9</td><td>56.8</td></tr><tr><td>✓</td><td>✓</td><td>64.2</td><td>70.0</td><td>63.4</td><td>65.9</td><td>63.9</td><td>58.6</td><td>66.7</td><td>57.5</td><td>60.9</td><td>57.4</td></tr></table>


Table A9. Ablation study of uncertainty-weighted mixup in Eq.14 and Eq. 15. on the AVE dataset.


<table><tr><td>Method</td><td>p^a_t, p^v_t</td><td>[ p^a_t ], [ p^v_t ]</td></tr><tr><td>Acc.(%)</td><td>80.3</td><td>80.6</td></tr></table>

derstanding. Coupled with our proposed class-balanced re-weighting strategy, the HAN model improves from 59.3 to60.4 for the event-level Type@AV. On the other hand, byintroducing the proposed uncertainty-aware mixup strategy,the event-level Type@AV increases from 59.3 to 60.

Ablation Study of the Uncertainty-weighted Mixup onthe AVE Dataset: As shown in Table A9, our experi-ments reveal that using $\lceil \hat { p } _ { t } ^ { a } \rceil$ and $\lceil \hat { p } _ { t } ^ { v } \rceil$ as the segment-levelpseudo labels instead of $\hat { p } _ { t } ^ { a }$ and $\hat { p } _ { t } ^ { v }$ for the uncertainty-weighted feature mixup strategy, in Eq. 14 and Eq. 15.in the main paper, results in a slightly better performanceon the AVE dataset.

# 15. Qualitative Results

Figures A6, A7 show event predictions of our method ver-sus competing baselines on sample videos from the LLPdataset [38]. Figure A8 shows the same, for sample videoson the AVE dataset [37]. As is evident from the figures,we see consistently accurate event-label predictions acrossdifferent videos, while also generally accurately localizingthem, the same is not the case for the baseline approaches.This feature is particularly prominent for instance, for thevisual event classes in the first video in Figure A7, or theaudio-visual events in the second video example in Fig-ure A8. However, there remain challenging scenarios wherealmost all methods struggle, such as the audio events in thefirst video example in Figure A7, which we hope to addressgoing forward.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/932be9411a7ed5ae3ea6ba56931f46b8ece3932e2e4df9d997bbc4aef0c553f6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/ba1868840edd103b87978372f6690666994d01868b3b31522d74b7216a3af40e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/c964b15b9988d4122f6c461a651e7c4074559dd327a25fda20f208b7e6635deb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/b4b3e173b33cb34a6b2364e108bf4109d05f64059db5de838b1e51d709d6c1d5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/c74f22143d45042e619bda630fea9ccfe0f146848b3718cbce2d272c12dc9630.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/14e22627a3d35b681abd1d73cac636a2d42b0cef8ed38634b9f9f8d9d742a25f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/8a90974c2bd59182f00e5cad3f967bbe03297e05e2973ca652caeba44bfefa60.jpg)



Figure A6. Comparison between predictions by UWAV and competing AVVP methods on the LLP dataset. “GT”: ground truth.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/04268989efca6173eb12b94f01a57af5f4148751215310aefa728e53e2bd1165.jpg)



Figure A7. Comparison between predictions by UWAV and competing AVVP methods on the LLP dataset. “GT”: ground truth.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-01/0a417743-9b9e-4880-8857-5c1d05e3e2b8/d1ac823abe6068641e287c46b77b881e14d90f1eb3bcaf1c254fdfd4998c2936.jpg)



Figure A8. Qualitative comparison between predictions by UWAV and previous methods on the AVE dataset. “GT”: ground truth.
