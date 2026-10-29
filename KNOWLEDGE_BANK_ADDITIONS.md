# Suggested Knowledge Bank additions

Statuses were checked in October 2026. Each entry names the sub-folder (category) to
put it in. Sub-folders become categories automatically.

**Legend.** 🟢 free and public (safe for a public repo) · 🟠 free to read, but check the
licence before redistributing · 🔴 paid or copyrighted (keep the repo private) · ⭐ highest value for your work

> Contact lenses and IOLs are **medical devices** (CDRH). The current Knowledge Bank is
> almost entirely drug/ICH guidance, so the device documents below fill the biggest gap.

## 1. Update what's already there

| Current file | Action |
|---|---|
| `FDA_Adaptive_2016.pdf` (draft) | ⭐ Replace with **FDA, Adaptive Designs for Clinical Trials of Drugs and Biologics (final, Nov 2019)** 🟢. For devices, also add **FDA/CDRH, Adaptive Designs for Medical Device Clinical Studies (final, 2016)** 🟢. |
| `FDA_PROclaims_2009.pdf` | Keep. Add **FDA PFDD Guidance 3, Selecting, Developing, or Modifying Fit-for-Purpose Clinical Outcome Assessments (final, 2025)** 🟢 and **FDA/CDRH, Principles for Selecting, Developing, Modifying, and Adapting PRO Instruments for Use in Medical Device Evaluation (final, 2022)** 🟢. |
| `FDA_ExternalControl_2023.pdf` | Still a draft (Feb 2023). Keep, and add the device-side **FDA/CDRH, Use of Real-World Evidence to Support Regulatory Decision-Making for Medical Devices (final, Dec 2025)** 🟢. |
| Scanned PDFs | Any file listed as skipped by `ingest.py` needs OCR, then re-ingest (see `SETUP.md`, step 9). |

## 2. Regulatory — devices and ophthalmic → `DeviceRegulatory/`, `Ophthalmic/`

| Document | Why | |
|---|---|---|
| ⭐ FDA/CDRH, **Design Considerations for Pivotal Clinical Investigations for Medical Devices** (final, 2013) | The device equivalent of ICH E9: controls, masking, bias, endpoints, sample size | 🟢 |
| ⭐ FDA/CDRH, **Guidance for the Use of Bayesian Statistics in Medical Device Clinical Trials** (2010) | Borrowing and priors in device trials; supports the external-control and BLIFT work | 🟢 |
| FDA, **Premarket Notification [510(k)] Guidance Document for Daily Wear Contact Lenses** (1994) and **…for Contact Lens Care Products** (1997) | The contact lens–specific FDA documents (old but still referenced) | 🟢 |
| FDA/CDRH, **Patient Preference Information** guidance (2016) and **Patient Engagement in the Design and Conduct of Medical Device Clinical Studies** (2022) | Patient-centred endpoints for devices | 🟢 |
| ⭐ **ISO 11980:2025**, Contact lenses and contact lens care products — guidance for clinical investigations | The reference standard for contact lens trial design (new 2025 edition) | 🔴 |
| **ISO 14155:2020** (+ amendments), Clinical investigation of medical devices — GCP | Device GCP: protocol and CSR content requirements | 🔴 |
| **ISO 18369-1…-4**, Contact lenses (vocabulary, tolerances, measurement, physicochemical) | Definitions and tolerances for lens parameters | 🔴 |
| **ISO 11979-7 / -10**, Intraocular lenses — clinical investigations | If the IOL side of J&J Vision is in scope | 🔴 |
| ⭐ Public **J&J Vision protocol/SAP PDFs on ClinicalTrials.gov** (e.g. NCT03059810, NCT04189224 "Prot_SAP" documents) | Real, public examples of contact lens protocols and SAPs (CLUE endpoints, crossover/contralateral designs). Very useful for the Check-a-section tab and for drafting | 🟢 |

## 3. Regulatory — ICH, FDA, EMA (drugs, methods) → `RegulatoryDocs/`

| Document | Status | |
|---|---|---|
| ⭐ ICH **E20** Adaptive Designs | Draft (Step 2b, 2025) | 🟢 |
| ICH **M11** Clinical Electronic Structured Harmonised Protocol (CeSHarP) + template | Final (Step 4, Nov 2025). Protocol structure; relevant to protocol drafting | 🟢 |
| ICH **E6(R3)** Good Clinical Practice | Final (2025) | 🟢 |
| ICH **E8(R1)** General Considerations for Clinical Studies | Final (2021) | 🟢 |
| ICH **E10** Choice of Control Group | Final | 🟢 |
| ICH **E3** Structure and Content of CSRs (+ Q&A) | Final; CSR writing | 🟢 |
| ⭐ FDA, **Adjusting for Covariates in Randomized Clinical Trials for Drugs and Biological Products** | Final (2023) | 🟢 |
| ⭐ FDA, **Non-Inferiority Clinical Trials to Establish Effectiveness** | Final (2016) | 🟢 |
| FDA, **Use of Bayesian Methodology in Clinical Trials of Drug and Biological Products** | Draft (Jan 2026) | 🟢 |
| FDA, **Interacting with FDA on Complex Innovative Trial Designs** | Final (2020) | 🟢 |
| FDA, **Master Protocols for Drug and Biological Product Development** | Check current status; the indexed 2022 file is the oncology-focused guidance | 🟢 |
| EMA, **Points to Consider on Multiplicity Issues** (2002, in effect) + **2017 draft guideline** | The 2017 draft is still not final | 🟢 |
| EMA, **Guideline on Adjustment for Baseline Covariates** (2015), **Missing Data in Confirmatory Trials** (2010), **Choice of a Non-Inferiority Margin** (2005) | Final | 🟢 |
| EMA, **Complex Clinical Trials – Questions and Answers** (2022) | Platform and master protocols | 🟢 |

## 4. Reporting guidelines → `ReportingGuidelines/`

| Document | |
|---|---|
| ⭐ **CONSORT 2025** and **SPIRIT 2025** statements | 🟢 |
| ⭐ **CONSORT extension for randomised crossover trials** (Dwan et al., BMJ 2019) | 🟢 |
| ⭐ Gamble et al., **Guidelines for the Content of Statistical Analysis Plans in Clinical Trials** (JAMA 2017) | 🟢 |
| CONSORT extension for non-inferiority and equivalence trials (JAMA 2012) | 🟢 |
| **CORE Reference** (Clarity and Openness in Reporting: E3-based), CSR authoring guide | 🟢 |

## 5. Ophthalmic science and outcomes → `Ophthalmic/`

| Document | |
|---|---|
| ⭐ Wirth et al., **Development of the Contact Lens User Experience: CLUE Scales**, Optom Vis Sci 2016;93(8):801–808 (open access, PMC4972477) | 🟢 |
| ⭐ **BJO "Ophthalmic statistics note" series** (Bunce et al., 2014 onward), starting with note 1 on the unit of analysis | 🟠 |
| ⭐ Ying, Maguire, Glynn, Rosner, **Tutorial on Biostatistics** papers: linear regression of correlated continuous eye data (2017) and correlated binary eye data (2018), Ophthalmic Epidemiol | 🟠 |
| Armstrong, **Statistical guidelines for the analysis of data obtained from one or both eyes**, Ophthalmic Physiol Opt 2013 | 🟠 |
| **TFOS International Workshop on Contact Lens Discomfort** (IOVS 2013, open access) | 🟢 |
| **BCLA CLEAR** reports (Contact Lens & Anterior Eye 2021, open access) | 🟢 |
| **TFOS DEWS III** (2025) | 🟢 |
| Chalmers et al., **CLDEQ-8** (Cont Lens Anterior Eye 2012) | 🟠 |

## 6. Statistics papers → `StatPapers/`

**Estimands and missing data**
- ⭐ Kahan et al., *The estimands framework: a primer on the ICH E9(R1) addendum*, BMJ 2024 🟢
- ⭐ National Research Council, *The Prevention and Treatment of Missing Data in Clinical Trials* (2010; free PDF) 🟢
- Carpenter, Roger & Kenward, *Reference-based imputation for longitudinal trials with protocol deviation*, J Biopharm Stat 2013 🔴

**Covariates and crossover**
- Kahan et al., *The risks and rewards of covariate adjustment in randomized trials*, Trials 2014 🟢
- ⭐ Kenward & Roger, *The use of baseline covariates in crossover studies*, Biostatistics 2010 🔴
- Kenward & Roger, *Small sample inference for fixed effects from restricted maximum likelihood*, Biometrics 1997 🔴

**Multiplicity**
- ⭐ Bretz et al., graphical approaches to sequentially rejective multiple testing (Stat Med 2009; Biom J 2011) 🟠
- Alosh, Bretz & Huque, *Advanced multiplicity adjustment methods in clinical trials*, Stat Med 2014 🔴

**Borrowing and external controls** (supports your methods research)
- ⭐ Viele et al., *Use of historical control data for assessing treatment effects in clinical trials*, Pharm Stat 2014 🟠
- ⭐ Schmidli et al., *Robust meta-analytic-predictive priors*, Biometrics 2014 🔴
- ⭐ Wang, Li et al. (FDA/CDRH), *Propensity score-integrated power prior approach for incorporating real-world evidence*, J Biopharm Stat 2019 🔴
- Ibrahim & Chen, *Power prior distributions for regression models*, Stat Sci 2000 🔴

**IPD-NMA with crossover and parallel trials** (supports the IPD-NMA papers)
- ⭐ Curtin, Elbourne & Altman, *Meta-analysis combining parallel and cross-over clinical trials* I–III, Stat Med 2002 🔴
- ⭐ Elbourne et al., *Meta-analyses involving cross-over trials: methodological issues*, Int J Epidemiol 2002 🟠
- NICE DSU TSD 18, *Population-adjusted indirect comparisons* (2016), and TSD 2–4 (evidence synthesis) 🟢
- Riley, Lambert & Abo-Zaid, *Meta-analysis of individual participant data*, BMJ 2010 🟢

**Platform trials**
- ⭐ Collignon et al., *Statistical considerations and regulatory perspectives on basket, umbrella and platform trials*, Clin Pharmacol Ther 2022 🟢
- Bofill Roig et al., *Model-based time trend adjustments in platform trials with non-concurrent controls*, BMC Med Res Methodol 2022 🟢

## 7. Textbooks → `StatBooks/` (🔴, keep the repo private)

| Book | Why |
|---|---|
| ⭐ Stroup, Ptukhina & Garai, *Generalized Linear Mixed Models* (2nd ed., CRC 2024) | GLMM / GLIMMIX theory and practice |
| ⭐ Stroup, Milliken, Claassen & Wolfinger, *SAS for Mixed Models* (2018) | PROC MIXED/GLIMMIX reference |
| ⭐ Senn, *Cross-over Trials in Clinical Research* (2nd ed.) | Complements Jones & Kenward |
| ⭐ Mallinckrodt et al., *Estimands, Estimators and Sensitivity Analysis in Clinical Trials* (CRC 2020) | Puts E9(R1) into practice |
| ⭐ Julious, *Sample Sizes for Clinical Trials* (2nd ed., CRC 2023) | Sample sizes for crossover, NI and equivalence designs |
| Dmitrienko, Tamhane & Bretz, *Multiple Testing Problems in Pharmaceutical Statistics* | Multiplicity |
| Rothmann, Wiens & Chan, *Design and Analysis of Non-Inferiority Trials* | NI margins |
| Riley, Tierney & Stewart, *Individual Participant Data Meta-Analysis* (Wiley 2021) | IPD-MA / NMA |
| Berry, Carlin, Lee & Müller, *Bayesian Adaptive Methods for Clinical Trials* | Bayesian designs |
| Senn, *Statistical Issues in Drug Development* (3rd ed., 2021) | Broad reference |

## After adding documents

1. `python ingest.py` (incremental), then read the skipped-files list at the end.
2. Add 1–3 questions per new document to `eval/questions.json` and run `python evaluate.py`.
3. Commit and push `index/` so the cloud app picks up the changes.
