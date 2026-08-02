#!/usr/bin/env python3
"""Rebuild cruxvault/raw/ for the CANDI literature wiki.

raw/ is gitignored: the sources are published papers under a mix of licences, several of
which permit reading but not redistribution, and the CANDI repo is public. This script
re-fetches every source from its open-access origin (Europe PMC / NCBI E-utilities /
Unpaywall / arXiv / bioRxiv / publisher OA) so a fresh clone can reconstruct the wiki's
evidence base.

    python3 cruxvault/tools/fetch_raw.py

Sources whose bytes differ from the ones originally ingested will show up as hash drift in
`crux validate` -- re-run `crux ingest` on those, then re-check the pages that cite them.

Not fetchable programmatically (publisher paywall / bot protection); add by hand:
  - Guo WL, Huang DS (2017) "An efficient method to transcription factor binding sites
    imputation via simultaneous completion of multiple matrices with positional
    consistency", Mol Biosyst 13(9):1827-37, doi:10.1039/c7mb00155j
    -> raw/guo-2017-tfbs-imputation-matrix-completion.pdf
  - Nix DA, Weigend AS (1994) "Estimating the mean and variance of the target
    probability distribution", IEEE ICNN, doi:10.1109/ICNN.1994.374138
    -> raw/nix-1994-nix-weigend.pdf
  - Harrell FE et al (1982) "Evaluating the yield of medical tests",
    JAMA 247(18):2543, doi:10.1001/jama.1982.03320430047030
    -> raw/harrell-1982-harrell-cindex.pdf
  - Czado C, Gneiting T, Held L (2009) "Predictive model assessment for count data",
    Biometrics 65(4):1254, doi:10.1111/j.1541-0420.2009.01191.x
    -> raw/czado-2009-czado-pit.pdf
  - Karlic R, Chung HR, Lasserre J, Vlahovicek K, Vingron M (2010) "Histone modification
    levels are predictive for gene expression", PNAS 107(7):2926,
    doi:10.1073/pnas.0909344107  (PNAS is bot-gated; PMC2814872 carries abstract only)
    -> raw/karlic-2010-histone-marks-expression.pdf
"""

ENTRIES = [
    # ---------------- The challenge paper itself (already on disk) ----------------
    dict(slug="schreiber-2023-encode-imputation-challenge",
         title="The ENCODE Imputation Challenge: a critical assessment of methods for cross-cell type imputation of epigenomic profiles",
         authors="Jacob M Schreiber, Carles A Boix, Jin Wook Lee, Hongyang Li, Yuanfang Guan, Chun-Chieh Chang, Jen-Chien Chang, Alex Hawkins-Hooker, Bernhard Scholkopf, Gabriele Schweikert, Mateo Rojas Carulla, Arif Canakoglu, Francesco Guzzo, Luca Nanni, Marco Masseroli, Mark James Carman, Pietro Pinoli, Chenyang Hong, Kevin Y Yip, Jeffrey P Spence, Sanjit Singh Batra, Yun S Song, Shaun Mahony, Zheng Zhang, Wuwei Tan, Yang Shen, Yuanfei Sun, Minyi Shi, Jessika Adrian, Richard S Sandstrom, Nina P Farrell, Jessica M Halow, Kristen Lee, Lixia Jiang, Xinqiong Yang, Charles B Epstein, J Seth Strattan, Bradley E Bernstein, Michael P Snyder, Manolis Kellis, William S Noble, Anshul Kundaje",
         year=2023, doi="10.1186/s13059-023-02915-y", origin="MC",
         ),

    # ---------------- Imputation methods (the direct prior art) ----------------
    dict(slug="ernst-2015-chromimpute",
         title="Large-scale imputation of epigenomic datasets for systematic annotation of diverse human tissues",
         authors="Jason Ernst, Manolis Kellis", year=2015, doi="10.1038/nbt.3157", origin="MC"),
    dict(slug="durham-2018-predictd",
         title="PREDICTD PaRallel Epigenomics Data Imputation with Cloud-based Tensor Decomposition",
         authors="Timothy J Durham, Maxwell W Libbrecht, J Jeffry Howbert, Jeff Bilmes, William Stafford Noble",
         year=2018, doi="10.1038/s41467-018-03635-9", origin="MC"),
    dict(slug="schreiber-2020-avocado",
         title="Avocado: a multi-scale deep tensor factorization method learns a latent representation of the human epigenome",
         authors="Jacob Schreiber, Timothy Durham, Jeffrey Bilmes, William Stafford Noble",
         year=2020, doi="10.1186/s13059-020-01977-6", origin="MC"),
    dict(slug="schreiber-2020-encode3-compendium",
         title="Completing the ENCODE3 compendium yields accurate imputations across a variety of assays and human biosamples",
         authors="Jacob Schreiber, Jeffrey Bilmes, William Stafford Noble",
         year=2020, doi="10.1186/s13059-020-01978-5", origin="C"),
    dict(slug="hawkins-hooker-2023-edice",
         title="Getting personal with epigenetics: towards individual-specific epigenomic imputation with machine learning",
         authors="Alex Hawkins-Hooker, Giovanni Visona, Tanmayee Narendra, Mateo Rojas-Carulla, Bernhard Scholkopf, Gabriele Schweikert",
         year=2023, doi="10.1038/s41467-023-40211-2", origin="M"),
    dict(slug="zhang-2023-epcot",
         title="A generalizable framework to comprehensively predict epigenome, chromatin organization, and transcriptome",
         authors="Zhenhao Zhang, Fan Feng, Yiyang Qiu, Jie Liu", year=2023,
         doi="10.1093/nar/gkad436", origin="M"),
    dict(slug="wen-2024-discriminative-histone-imputation",
         title="Discriminative histone imputation using chromatin accessibility",
         authors="Wen Wen, Jiaxin Zhong, Zhaoxi Zhang, Lijuan Jia, Tinyi Chu, Nating Wang, Charles G Danko, Zhong Wang",
         year=2024, doi=None, origin="M", biorxiv=True),
    dict(slug="guo-2017-tfbs-imputation-matrix-completion",
         title="An efficient method to transcription factor binding sites imputation via simultaneous completion of multiple matrices with positional consistency",
         authors="Wei-Li Guo, De-Shuang Huang", year=2017, doi="10.1039/c7mb00155j", origin="C"),
    dict(slug="qin-2017-tf-binding-deep-learning-imputation",
         title="Imputation for transcription factor binding predictions based on deep learning",
         authors="Qian Qin, Jianxing Feng", year=2017, doi="10.1371/journal.pcbi.1005403", origin="C"),
    dict(slug="schreiber-2020-pitfall-cross-cell-type",
         title="A pitfall for machine learning methods aiming to predict across cell types",
         authors="Jacob Schreiber, Ritambhara Singh, Jeffrey Bilmes, William Stafford Noble",
         year=2020, doi="10.1186/s13059-020-02177-y", origin="C"),

    # ---------------- Reference epigenome resources / consortia ----------------
    dict(slug="roadmap-2015-111-reference-epigenomes",
         title="Integrative analysis of 111 reference human epigenomes",
         authors="Roadmap Epigenomics Consortium, Anshul Kundaje, Wouter Meuleman, Jason Ernst, Misha Bilenky, Angela Yen, et al",
         year=2015, doi="10.1038/nature14248", origin="C"),
    dict(slug="encode-2020-expanded-encyclopedias",
         title="Expanded encyclopaedias of DNA elements in the human and mouse genomes",
         authors="ENCODE Project Consortium, Jill E Moore, Michael J Purcaro, Henry E Pratt, Charles B Epstein, Noam Shoresh, Jessika Adrian, Trupti Kawli, Carrie A Davis, Alexander Dobin, et al",
         year=2020, doi="10.1038/s41586-020-2493-4", origin="MC"),
    dict(slug="stunnenberg-2016-ihec-blueprint",
         title="The International Human Epigenome Consortium: A Blueprint for Scientific Collaboration and Discovery",
         authors="Hendrik G Stunnenberg, International Human Epigenome Consortium, Martin Hirst",
         year=2016, doi="10.1016/j.cell.2016.11.007", origin="C"),
    dict(slug="bujold-2016-ihec-data-portal",
         title="The International Human Epigenome Consortium Data Portal",
         authors="David Bujold, David Anderson de Lima Morais, Carol Gauthier, Catherine Cote, Maxime Caron, Tony Kwan, Kuang Chung Chen, Jonathan Laperle, Alexei Nordell Markovits, Tomi Pastinen, et al",
         year=2016, doi="10.1016/j.cels.2016.10.019", origin="M"),
    dict(slug="ramilowski-2020-lncrna-functional-annotation",
         title="Functional annotation of human long noncoding RNAs via molecular phenotyping",
         authors="Jordan A Ramilowski, Chi Wai Yip, Saumya Agrawal, Jen-Chien Chang, Yari Ciani, Ivan V Kulakovskiy, et al",
         year=2020, doi="10.1101/gr.254219.119", origin="C"),
    dict(slug="gtex-2017-genetic-effects-gene-expression",
         title="Genetic effects on gene expression across human tissues",
         authors="GTEx Consortium, Laboratory Data Analysis and Coordinating Center (LDACC), Statistical Methods groups, Enhancing GTEx (eGTEx) groups, NIH Common Fund, NIH/NCI, et al",
         year=2017, doi="10.1038/nature24277", origin="C"),
    dict(slug="lindeboom-2021-human-cell-atlas",
         title="Towards a Human Cell Atlas: Taking Notes from the Past",
         authors="Rik G H Lindeboom, Aviv Regev, Sarah A Teichmann", year=2021,
         doi="10.1016/j.tig.2021.03.007", origin="C"),
    dict(slug="boix-2021-regulatory-genomic-circuitry",
         title="Regulatory genomic circuitry of human disease loci by integrative epigenomics",
         authors="Carles A Boix, Benjamin T James, Yongjin P Park, Wouter Meuleman, Manolis Kellis",
         year=2021, doi="10.1038/s41586-020-03145-z", origin="C"),
    dict(slug="harrow-2012-gencode",
         title="GENCODE: the reference human genome annotation for The ENCODE Project",
         authors="Jennifer Harrow, Adam Frankish, Jose M Gonzalez, Electra Tapanari, Mark Diekhans, Felix Kokocinski, et al",
         year=2012, doi="10.1101/gr.135350.111", origin="C"),
    dict(slug="fantom5-2014-promoter-level-expression-atlas",
         title="A promoter-level mammalian expression atlas",
         authors="FANTOM Consortium and the RIKEN PMI and CLST (DGT), Alistair R R Forrest, Hideya Kawaji, Michael Rehli, J Kenneth Baillie, Michiel J L de Hoon, et al",
         year=2014, doi="10.1038/nature13182", origin="C"),

    # ---------------- Assay processing, QC, peak calling ----------------
    dict(slug="zhang-2008-macs",
         title="Model-based Analysis of ChIP-Seq (MACS)",
         authors="Yong Zhang, Tao Liu, Clifford A Meyer, Jerome Eeckhoute, David S Johnson, Bradley E Bernstein, Chad Nusbaum, Richard M Myers, Myles Brown, Wei Li, X Shirley Liu",
         year=2008, doi="10.1186/gb-2008-9-9-r137", origin="MC"),
    dict(slug="landt-2012-chip-seq-guidelines",
         title="ChIP-seq guidelines and practices of the ENCODE and modENCODE consortia",
         authors="Stephen G Landt, Georgi K Marinov, Anshul Kundaje, Pouya Kheradpour, Florencia Pauli, Serafim Batzoglou, Bradley E Bernstein, Peter Bickel, James B Brown, Philip Cayting, et al",
         year=2012, doi="10.1101/gr.136184.111", origin="M"),
    dict(slug="amemiya-2019-encode-blacklist",
         title="The ENCODE Blacklist: Identification of Problematic Regions of the Genome",
         authors="Haley M Amemiya, Anshul Kundaje, Alan P Boyle", year=2019,
         doi="10.1038/s41598-019-45839-z", origin="C"),
    dict(slug="jung-2014-sequencing-depth-chip-seq",
         title="Impact of sequencing depth in ChIP-seq experiments",
         authors="Youngsook L Jung, Lovelace J Luquette, Joshua W K Ho, Francesco Ferrari, Michael Tolstorukov, Aki Minoda, et al",
         year=2014, doi="10.1093/nar/gku178", origin="C"),
    dict(slug="teng-2021-chip-seq-batch-effects",
         title="Characterizing batch effects and binding site-specific variability in ChIP-seq data",
         authors="Mingxiang Teng, Dongliang Du, Danfeng Chen, Rafael A Irizarry", year=2021,
         doi="10.1093/nargab/lqab098", origin="M"),
    dict(slug="langmead-2012-bowtie2",
         title="Fast gapped-read alignment with Bowtie 2",
         authors="Ben Langmead, Steven L Salzberg", year=2012, doi="10.1038/nmeth.1923", origin="C"),
    dict(slug="li-2009-bwa",
         title="Fast and accurate short read alignment with Burrows-Wheeler transform",
         authors="Heng Li, Richard Durbin", year=2009, doi="10.1093/bioinformatics/btp324", origin="C"),
    dict(slug="mckenna-2010-gatk",
         title="The Genome Analysis Toolkit: a MapReduce framework for analyzing next-generation DNA sequencing data",
         authors="Aaron McKenna, Matthew Hanna, Eric Banks, Andrey Sivachenko, Kristian Cibulskis, Andrew Kernytsky, et al",
         year=2010, doi="10.1101/gr.107524.110", origin="C"),

    # ---------------- Normalization (the challenge's central theme) ----------------
    dict(slug="xiang-2020-s3norm",
         title="S3norm: simultaneous normalization of sequencing depth and signal-to-noise ratio in epigenomic data",
         authors="Guanjue Xiang, Cheryl A Keller, Belinda Giardine, Lin An, Qunhua Li, Yu Zhang, Ross C Hardison",
         year=2020, doi="10.1093/nar/gkaa105", origin="MC"),
    dict(slug="zhao-2020-quantile-normalization-correctly",
         title="How to do quantile normalization correctly for gene expression data analyses",
         authors="Yilin Zhao, Limsoon Wong, Wilson Wen Bin Goh", year=2020,
         doi="10.1038/s41598-020-72664-6", origin="C"),
    dict(slug="townes-2020-quantile-normalization-scrnaseq",
         title="Quantile normalization of single-cell RNA-seq read counts without unique molecular identifiers",
         authors="F William Townes, Rafael A Irizarry", year=2020,
         doi="10.1186/s13059-020-02078-0", origin="C"),
    dict(slug="bonhoure-2014-chip-seq-spiking",
         title="Quantifying ChIP-seq data: a spiking method providing an internal reference for sample-to-sample normalization",
         authors="Nicolas Bonhoure, Gergana Bounova, David Bernasconi, Viviane Praz, Frederic Lammers, Donatella Canella, et al",
         year=2014, doi="10.1101/gr.168260.113", origin="C"),
    dict(slug="polit-2021-chipin",
         title="CHIPIN: ChIP-seq inter-sample normalization based on signal invariance across transcriptionally constant genes",
         authors="Lelia Polit, Gwenneg Kerdivel, Sebastian Gregoricchio, Michela Esposito, Christel Guillouf, Valentina Boeva",
         year=2021, doi="10.1186/s12859-021-04320-3", origin="C"),
    dict(slug="reske-2020-atac-seq-normalization",
         title="ATAC-seq normalization method can significantly affect differential accessibility analysis and interpretation",
         authors="Jake J Reske, Mike R Wilson, Ronald L Chandler", year=2020,
         doi="10.1186/s13072-020-00342-y", origin="C"),
    dict(slug="hicks-2018-smooth-quantile-normalization",
         title="Smooth quantile normalization",
         authors="Stephanie C Hicks, Kwame Okrah, Joseph N Paulson, John Quackenbush, Rafael A Irizarry, Hector Corrada Bravo",
         year=2018, doi="10.1093/biostatistics/kxx028", origin="C"),
    dict(slug="angelini-2015-chip-seq-normalization-diagnostic",
         title="Is this the right normalization? A diagnostic tool for ChIP-seq normalization",
         authors="Claudia Angelini, Ruth Heller, Rita Volkinshtein, Daniel Yekutieli", year=2015,
         doi="10.1186/s12859-015-0579-z", origin="C"),
    dict(slug="anders-2010-deseq",
         title="Differential expression analysis for sequence count data",
         authors="Simon Anders, Wolfgang Huber", year=2010, doi="10.1186/gb-2010-11-10-r106", origin="M"),

    # ---------------- Chromatin state annotation (downstream use) ----------------
    dict(slug="hoffman-2012-segway",
         title="Unsupervised pattern discovery in human chromatin structure through genomic segmentation",
         authors="Michael M Hoffman, Orion J Buske, Jie Wang, Zhiping Weng, Jeff A Bilmes, William Stafford Noble",
         year=2012, doi="10.1038/nmeth.1937", origin="M"),
    dict(slug="shahraki-2024-robust-chromatin-state-annotation",
         title="Robust chromatin state annotation",
         authors="Mehdi Foroozandeh Shahraki, Marjan Farahbod, Maxwell W Libbrecht", year=2024,
         doi="10.1101/gr.278726.123", origin="M"),
    dict(slug="ernst-2012-chromhmm", title="ChromHMM: automating chromatin-state discovery and characterization",
         authors="Jason Ernst, Manolis Kellis", year=2012, doi="10.1038/nmeth.1906", origin="X"),

    # ---------------- Sequence-based deep learning models ----------------
    dict(slug="avsec-2021-enformer",
         title="Effective gene expression prediction from sequence by integrating long-range interactions",
         authors="Ziga Avsec, Vikram Agarwal, Daniel Visentin, Joseph R Ledsam, Agnieszka Grabska-Barwinska, Kyle R Taylor, Yannis Assael, John Jumper, Pushmeet Kohli, David R Kelley",
         year=2021, doi="10.1038/s41592-021-01252-x", origin="M"),
    dict(slug="avsec-2021-bpnet",
         title="Base-resolution models of transcription-factor binding reveal soft motif syntax",
         authors="Ziga Avsec, Melanie Weilert, Avanti Shrikumar, Sabrina Krueger, Amr Alexandari, Khyati Dalal, et al",
         year=2021, doi="10.1038/s41588-021-00782-6", origin="C"),
    dict(slug="ji-2021-dnabert",
         title="DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome",
         authors="Yanrong Ji, Zhihan Zhou, Han Liu, Ramana V Davuluri", year=2021,
         doi="10.1093/bioinformatics/btab083", origin="M"),

    # ---------------- ML architecture & self-supervision ----------------
    dict(slug="devlin-2019-bert",
         title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
         authors="Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova", year=2019,
         arxiv="1810.04805", origin="M"),
    dict(slug="perez-2018-film",
         title="FiLM: Visual Reasoning with a General Conditioning Layer",
         authors="Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville",
         year=2018, arxiv="1709.07871", origin="M"),
    dict(slug="yoon-2020-vime",
         title="VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain",
         authors="Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar", year=2020,
         url="https://proceedings.neurips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf",
         origin="M"),
    dict(slug="vaswani-2017-attention-is-all-you-need",
         title="Attention Is All You Need",
         authors="Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, Illia Polosukhin",
         year=2017, arxiv="1706.03762", origin="X"),
    dict(slug="su-2021-roformer-rope",
         title="RoFormer: Enhanced Transformer with Rotary Position Embedding",
         authors="Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu", year=2021,
         arxiv="2104.09864", origin="X"),
    dict(slug="he-2022-masked-autoencoders",
         title="Masked Autoencoders Are Scalable Vision Learners",
         authors="Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, Ross Girshick", year=2022,
         arxiv="2111.06377", origin="X"),
    dict(slug="assran-2023-ijepa",
         title="Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture",
         authors="Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas",
         year=2023, arxiv="2301.08243", origin="X"),
    dict(slug="guo-2017-calibration-modern-neural-networks",
         title="On Calibration of Modern Neural Networks",
         authors="Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q Weinberger", year=2017,
         arxiv="1706.04599", origin="X"),

    # ================= round 2: tiers 1-3 from the scout consolidation =================
    dict(slug="ahlmann-eltze-2023",
         title="Comparison of transformations for single-cell RNA-seq data",
         authors="Constantin Ahlmann-Eltze, Wolfgang Huber",
         year=2023, doi="10.1038/s41592-023-01814-1", origin="R2-T3"),
    dict(slug="aksu-2026-corgi",
         title="Context-aware sequence-to-function model of human gene regulation",
         authors="Ekin Deniz Aksu, Martin Vingron",
         year=2026, doi="10.1038/s41467-026-75527-2", origin="R2-T3"),
    dict(slug="ashuach-2022-peakvi",
         title="PeakVI: A deep generative model for single-cell chromatin accessibility analysis",
         authors="Tal Ashuach, Daniel A. Reidenbach, Adam Gayoso, Nir Yosef",
         year=2022, doi="10.1016/j.crmeth.2022.100182", origin="R2-T3"),
    dict(slug="ashuach-2023-multivi",
         title="MultiVI: deep generative model for the integration of multimodal data",
         authors="Tal Ashuach, Mariano I. Gabitto, Rohan V. Koodli, Giuseppe-Antonio Saldi, Michael I. Jordan, Nir Yosef",
         year=2023, doi="10.1038/s41592-023-01909-9", origin="R2-T3"),
    dict(slug="avsec-2026-alphagenome",
         title="Advancing regulatory variant effect prediction with AlphaGenome",
         authors="Ziga Avsec, Natasha Latysheva, Jun Cheng, Guido Novati, Kyle R. Taylor, Tom Ward, Clare Bycroft, Lauren Nicolaisen, Eirini Arvaniti, Joshua Pan, Raina Thomas, Vincent Dutordoir, et al",
         year=2026, doi="10.1038/s41586-025-10014-0", origin="R2-T3"),
    dict(slug="aygun-2025-era",
         title="An AI system to help scientists write expert-level empirical software",
         authors="Eser Aygun, Anastasiya Belyaeva, Gheorghe Comanici, Marc Coram, Hao Cui, Jake Garrison, Renee Johnston Anton Kast, Cory Y. McLean, Peter Norgaard, Zahra Shamsi, David Smalling, James Thompson, et al",
         year=2025, arxiv="2509.06503", origin="R2-T1"),
    dict(slug="balestriero-2025-lejepa",
         title="LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics",
         authors="Randall Balestriero, Yann LeCun",
         year=2025, arxiv="2511.08544", origin="R2-T1"),
    dict(slug="barbadilla-martinez-2025",
         title="Predicting gene expression from DNA sequence using deep learning models",
         authors="Lucia Barbadilla-Martinez, Noud Klaassen, Bas van Steensel, Jeroen de Ridder",
         year=2025, doi="10.1038/s41576-025-00841-2", origin="R2-T3"),
    dict(slug="bardes-2021-vicreg",
         title="VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning",
         authors="Adrien Bardes, Jean Ponce, Yann LeCun",
         year=2021, arxiv="2105.04906", origin="R2-T2"),
    dict(slug="boshar-2025-ntv3",
         title="A foundational model for joint sequence-function multi-species modeling at scale for long-range genomic prediction",
         authors="Sam Boshar, Benjamin Evans, Ziqi Tang, Armand Picard, Yanis Adel, Franziska K Lorbeer, Chandana Rajesh, Tristan Karch, Shawn Sidbon, David Emms, Javier Mendoza-Revilla, Fatimah Al-Ani, et al",
         year=2025, doi="10.64898/2025.12.22.695963", origin="R2-T3"),
    dict(slug="boyeau-2025-mrvi",
         title="Deep generative modeling of sample-level heterogeneity in single-cell genomics",
         authors="Pierre Boyeau, Justin Hong, Adam Gayoso, Martin Kim, Jose L. McFaline-Figueroa, Michael I. Jordan, Elham Azizi, Can Ergen, Nir Yosef",
         year=2025, doi="10.1038/s41592-025-02808-x", origin="R2-T3"),
    dict(slug="brixi-2026-evo2",
         title="Genome modelling and design across all domains of life with Evo 2",
         authors="Garyk Brixi, Matthew G. Durrant, Jerome Ku, Mohsen Naghipourfar, Michael Poli, Gwanggyu Sun, Greg Brockman, Daniel Chang, Alison Fanton, Gabriel A. Gonzalez, Samuel H. King, David B. Li, et al",
         year=2026, doi="10.1038/s41586-026-10176-5", origin="R2-T3"),
    dict(slug="cameron-2008-cgm-bootstrap",
         title="Bootstrap-Based Improvements for Inference with Clustered Errors",
         authors="A. Colin Cameron, Jonah B. Gelbach, Douglas L. Miller",
         year=2008, doi="10.1162/rest.90.3.414", origin="R2-T2"),
    dict(slug="chen-2019-dynconv",
         title="Dynamic Convolution: Attention over Convolution Kernels",
         authors="Yinpeng Chen, Xiyang Dai, Mengchen Liu, Dongdong Chen, Lu Yuan, Zicheng Liu",
         year=2019, arxiv="1912.03458", origin="R2-T1"),
    dict(slug="chen-2022-sei",
         title="A sequence-based global map of regulatory activity for deciphering human genetics",
         authors="Kathleen M. Chen, Aaron K. Wong, Olga G. Troyanskaya, Jian Zhou",
         year=2022, doi="10.1038/s41588-022-01102-2", origin="R2-T3"),
    dict(slug="chen-2025-epiagent",
         title="EpiAgent: foundation model for single-cell epigenomics",
         authors="Xiaoyang Chen, Keyi Li, Xuejian Cui, Zian Wang, Qun Jiang, Jiacheng Lin, Zhen Li, Zijing Gao, Hairong Lv, Rui Jiang",
         year=2025, doi="10.1038/s41592-025-02822-z", origin="R2-T3"),
    dict(slug="choudhary-2022-sctransform-v2",
         title="Comparison and evaluation of statistical error models for scRNA-seq",
         authors="Saket Choudhary, Rahul Satija",
         year=2022, doi="10.1186/s13059-021-02584-9", origin="R2-T2"),
    dict(slug="czado-2009-czado-pit",
         title="Predictive Model Assessment for Count Data",
         authors="Claudia Czado, Tilmann Gneiting, Leonhard Held",
         year=2009, doi="10.1111/j.1541-0420.2009.01191.x", origin="R2-T1"),
    dict(slug="dieng-2018-dieng-skip",
         title="Avoiding Latent Variable Collapse With Generative Skip Models",
         authors="Adji B. Dieng, Yoon Kim, Alexander M. Rush, David M. Blei",
         year=2018, arxiv="1807.04863", origin="R2-T2"),
    dict(slug="elazar-2020-amnesic",
         title="Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals",
         authors="Yanai Elazar, Shauli Ravfogel, Alon Jacovi, Yoav Goldberg",
         year=2020, arxiv="2006.00995", origin="R2-T2"),
    dict(slug="eraslan-2019-dca",
         title="Single-cell RNA-seq denoising using a deep count autoencoder",
         authors="Gokcen Eraslan, Lukas M. Simon, Maria Mircea, Nikola S. Mueller, Fabian J. Theis",
         year=2019, doi="10.1038/s41467-018-07931-2", origin="R2-T1"),
    dict(slug="fu-2025-get",
         title="A foundation model of transcription across human cell types",
         authors="Xi Fu, Shentong Mo, Alejandro Buendia, Anouchka P. Laurent, Anqi Shao, Maria del Mar Alvarez-Torres, Tianji Yu, Jimin Tan, Jiayu Su, Romella Sagatelian, Adolfo A. Ferrando, Alberto Ciccia, et al",
         year=2025, doi="10.1038/s41586-024-08391-z", origin="R2-T3"),
    dict(slug="gao-2024-epigept",
         title="EpiGePT: a pretrained transformer-based language model for context-specific human epigenomics",
         authors="Zijing Gao, Qiao Liu, Wanwen Zeng, Rui Jiang, Wing Hung Wong",
         year=2024, doi="10.1186/s13059-024-03449-7", origin="R2-T3"),
    dict(slug="garnelo-2018-cnp",
         title="Conditional Neural Processes",
         authors="Marta Garnelo, Dan Rosenbaum, Chris J. Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo J. Rezende, S. M. Ali Eslami",
         year=2018, arxiv="1807.01613", origin="R2-T1"),
    dict(slug="garrido-2022-rankme",
         title="RankMe: Assessing the downstream performance of pretrained self-supervised representations by their rank",
         authors="Quentin Garrido, Randall Balestriero, Laurent Najman, Yann Lecun",
         year=2022, arxiv="2210.02885", origin="R2-T2"),
    dict(slug="garrido-2024-iwm",
         title="Learning and Leveraging World Models in Visual Representation Learning",
         authors="Quentin Garrido, Mahmoud Assran, Nicolas Ballas, Adrien Bardes, Laurent Najman, Yann LeCun",
         year=2024, arxiv="2403.00504", origin="R2-T2"),
    dict(slug="gneiting-2007-gneiting-raftery",
         title="Strictly Proper Scoring Rules, Prediction, and Estimation",
         authors="Tilmann Gneiting, Adrian E Raftery",
         year=2007, doi="10.1198/016214506000001437", origin="R2-T1"),
    dict(slug="hafemeister-2019-sctransform",
         title="Normalization and variance stabilization of single-cell RNA-seq data using regularized negative binomial regression",
         authors="Christoph Hafemeister, Rahul Satija",
         year=2019, doi="10.1186/s13059-019-1874-1", origin="R2-T2"),
    dict(slug="harrell-1982-harrell-cindex",
         title="Evaluating the Yield of Medical Tests",
         authors="Frank E. Harrell",
         year=1982, doi="10.1001/jama.1982.03320430047030", origin="R2-T1"),
    dict(slug="hingerl-2025-scooby",
         title="scooby: modeling multimodal genomic profiles from DNA sequence at single-cell resolution",
         authors="Johannes C. Hingerl, Laura D. Martens, Alexander Karollus, Trevor Manz, Jason D. Buenrostro, Fabian J. Theis, Julien Gagneur",
         year=2025, doi="10.1038/s41592-025-02854-5", origin="R2-T3"),
    dict(slug="ho-2019-axial",
         title="Axial Attention in Multidimensional Transformers",
         authors="Jonathan Ho, Nal Kalchbrenner, Dirk Weissenborn, Tim Salimans",
         year=2019, arxiv="1912.12180", origin="R2-T1"),
    dict(slug="ho-2022-cfg",
         title="Classifier-Free Diffusion Guidance",
         authors="Jonathan Ho, Tim Salimans",
         year=2022, arxiv="2207.12598", origin="R2-T2"),
    dict(slug="javed-2025-epibert",
         title="A multi-modal transformer for cell type-agnostic regulatory predictions",
         authors="Nauman Javed, Thomas Weingarten, Arijit Sehanobish, Adam Roberts, Avinava Dubey, Krzysztof Choromanski, Bradley E. Bernstein",
         year=2025, doi="10.1016/j.xgen.2025.100762", origin="R2-T3"),
    dict(slug="karimzadeh-2018-umap-mappability",
         title="Umap and Bismap: quantifying genome and methylome mappability",
         authors="Mehran Karimzadeh, Carl Ernst, Anshul Kundaje, Michael M Hoffman",
         year=2018, doi="10.1093/nar/gky677", origin="R2-T2"),
    dict(slug="karollus-2023",
         title="Current sequence-based models capture gene expression determinants in promoters but mostly ignore distal enhancers",
         authors="Alexander Karollus, Thomas Mauermeier, Julien Gagneur",
         year=2023, doi="10.1186/s13059-023-02899-9", origin="R2-T3"),
    dict(slug="karras-2019-stylegan2",
         title="Analyzing and Improving the Image Quality of StyleGAN",
         authors="Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila",
         year=2019, arxiv="1912.04958", origin="R2-T2"),
    dict(slug="kelley-2018-basenji",
         title="Sequential regulatory activity prediction across chromosomes with convolutional neural networks",
         authors="David R. Kelley, Yakir A. Reshef, Maxwell Bileschi, David Belanger, Cory Y. McLean, Jasper Snoek",
         year=2018, doi="10.1101/gr.227819.117", origin="R2-T3"),
    dict(slug="kendall-2017-kendall-uw",
         title="Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics",
         authors="Alex Kendall, Yarin Gal, Roberto Cipolla",
         year=2017, arxiv="1705.07115", origin="R2-T1"),
    dict(slug="kingma-2013-vae",
         title="Auto-Encoding Variational Bayes",
         authors="Diederik P Kingma, Max Welling",
         year=2013, arxiv="1312.6114", origin="R2-T1"),
    dict(slug="koh-2017-coda",
         title="Denoising genome-wide histone ChIP-seq with convolutional neural networks",
         authors="Pang Wei Koh, Emma Pierson, Anshul Kundaje",
         year=2017, doi="10.1093/bioinformatics/btx243", origin="R2-T3"),
    dict(slug="kuleshov-2018",
         title="Accurate Uncertainties for Deep Learning Using Calibrated Regression",
         authors="Volodymyr Kuleshov, Nathan Fenner, Stefano Ermon",
         year=2018, arxiv="1807.00263", origin="R2-T2"),
    dict(slug="lal-2021-atacworks",
         title="Deep learning-based enhancement of epigenomics data with AtacWorks",
         authors="Avantika Lal, Zachary D. Chiang, Nikolai Yakovenko, Fabiana M. Duarte, Johnny Israeli, Jason D. Buenrostro",
         year=2021, doi="10.1038/s41467-021-21765-5", origin="R2-T3"),
    dict(slug="lal-2025-grelu",
         title="gReLU: a comprehensive framework for DNA sequence modeling and design",
         authors="Avantika Lal, Laura Gunsalus, Surag Nair, Tommaso Biancalani, Gokcen Eraslan",
         year=2025, doi="10.1038/s41592-025-02868-z", origin="R2-T3"),
    dict(slug="lal-2026-decima",
         title="Decoding sequence determinants of gene expression in diverse cellular and disease states",
         authors="Avantika Lal, Alexander Karollus, Laura Gunsalus, David Garfield, Surag Nair, Alex M. Tseng, M. Grace Gordon, John Blischak, Bryce Van De Geijn, Tushar Bhangale, Jenna L. Collier, Nathaniel Diamant, et al",
         year=2026, doi="10.1038/s41592-026-03102-0", origin="R2-T3"),
    dict(slug="lee-2018-set-transformer",
         title="Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks",
         authors="Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, Yee Whye Teh",
         year=2018, arxiv="1810.00825", origin="R2-T1"),
    dict(slug="leek-2010-leek-batch",
         title="Tackling the widespread and critical impact of batch effects in high-throughput data",
         authors="Jeffrey T. Leek, Robert B. Scharpf, Hector Corrada Bravo, David Simcha, Benjamin Langmead, W. Evan Johnson, Donald Geman, Keith Baggerly, Rafael A. Irizarry",
         year=2010, doi="10.1038/nrg2825", origin="R2-T2"),
    dict(slug="li-2011-idr",
         title="Measuring reproducibility of high-throughput experiments",
         authors="Qunhua Li, James B. Brown, Haiyan Huang, Peter J. Bickel",
         year=2011, doi="10.1214/11-AOAS466", origin="R2-T1"),
    dict(slug="lin-2025-epiverse",
         title="Unveiling chromatin dynamics with virtual epigenome",
         authors="Ming-Yu Lin, Yu-Cheng Lo, Jui-Hung Hung",
         year=2025, doi="10.1038/s41467-025-58481-3", origin="R2-T3"),
    dict(slug="linder-2025-borzoi",
         title="Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation",
         authors="Johannes Linder, Divyanshi Srivastava, Han Yuan, Vikram Agarwal, David R. Kelley",
         year=2025, doi="10.1038/s41588-024-02053-6", origin="R2-T3"),
    dict(slug="lopez-2018-scvi",
         title="Deep generative modeling for single-cell transcriptomics",
         authors="Romain Lopez, Jeffrey Regier, Michael B. Cole, Michael I. Jordan, Nir Yosef",
         year=2018, doi="10.1038/s41592-018-0229-2", origin="R2-T1"),
    dict(slug="loshchilov-2016-sgdr",
         title="SGDR: Stochastic Gradient Descent with Warm Restarts",
         authors="Ilya Loshchilov, Frank Hutter",
         year=2016, arxiv="1608.03983", origin="R2-T1"),
    dict(slug="loshchilov-2017-adamw",
         title="Decoupled Weight Decay Regularization",
         authors="Ilya Loshchilov, Frank Hutter",
         year=2017, arxiv="1711.05101", origin="R2-T1"),
    dict(slug="lotfollahi-2023-cpa",
         title="Predicting cellular responses to complex perturbations in highthroughput screens",
         authors="Mohammad Lotfollahi, Anna Klimovskaia Susmelj, Carlo De Donno, Leon Hetzel, Yuge Ji, Ignacio L Ibarra, Sanjay R Srivatsan, Mohsen Naghipourfar, Riza M Daza, Beth Martin, Jay Shendure, Jose L McFalineFigueroa, et al",
         year=2023, doi="10.15252/msb.202211517", origin="R2-T2"),
    dict(slug="martens-2023",
         title="Modeling fragment counts improves single-cell ATAC-seq analysis",
         authors="Laura D. Martens, David S. Fischer, Vicente A. Yepez, Fabian J. Theis, Julien Gagneur",
         year=2023, doi="10.1038/s41592-023-02112-6", origin="R2-T3"),
    dict(slug="mitra-2023-structured-missing",
         title="Learning from data with structured missingness",
         authors="Robin Mitra, Sarah F. McGough, Tapabrata Chakraborti, Chris Holmes, Ryan Copping, Niels Hagenbuch, Stefanie Biedermann, Jack Noonan, Brieuc Lehmann, Aditi Shenvi, Xuan Vinh Doan, David Leslie, et al",
         year=2023, doi="10.1038/s42256-022-00596-z", origin="R2-T2"),
    dict(slug="mller-2025-deepdive",
         title="Disentangling covariate effects on single cell-resolved epigenomes with DeepDive",
         authors="Andreas Fnss Mller, Jesper Grud Skat Madsen",
         year=2025, doi="10.1101/2025.09.30.679466", origin="R2-T2"),
    dict(slug="moore-2026-ccre-v4",
         title="An expanded registry of candidate cis-regulatory elements",
         authors="Jill E. Moore, Henry E. Pratt, Kaili Fan, Nishigandha Phalke, Jonathan Fisher, Shaimae I. Elhajjajy, Gregory Andrews, Mingshi Gao, Nicole Shedd, Yu Fu, Matthew C. Lacadie, Jair Meza, et al",
         year=2026, doi="10.1038/s41586-025-09909-9", origin="R2-T3"),
    dict(slug="mostafavi-2026-modality-gap",
         title="A modality gap in personal-genome prediction by sequence-to-function models",
         authors="Sara Mostafavi, Xinming Tu, Anna Spiro, Maria Chikina",
         year=2026, doi="10.64898/2026.02.01.702969", origin="R2-T3"),
    dict(slug="murphy-2024-enformer-celltyping",
         title="Predicting cell type-specific epigenomic profiles accounting for distal genetic effects",
         authors="Alan E. Murphy, William Beardall, Marek Rei, Mike Phuycharoen, Nathan G. Skene",
         year=2024, doi="10.1038/s41467-024-54441-5", origin="R2-T3"),
    dict(slug="neufeld-2023-thinning",
         title="Data thinning for convolution-closed distributions",
         authors="Anna Neufeld, Ameer Dharamshi, Lucy L. Gao, Daniela Witten",
         year=2023, arxiv="2301.07276", origin="R2-T2"),
    dict(slug="nix-1994-nix-weigend",
         title="Estimating the mean and variance of the target probability distribution",
         authors="D.A. Nix, A.S. Weigend",
         year=1994, doi="10.1109/ICNN.1994.374138", origin="R2-T1"),
    dict(slug="pampari-2024-chrombpnet",
         title="ChromBPNet: bias factorized, base-resolution deep learning models of chromatin accessibility reveal cis-regulatory sequence syntax, transcription factor footprints and regulatory variants",
         authors="Anusri Pampari, Anna Shcherbina, Evgeny Z. Kvon, Michael Kosicki, Surag Nair, Soumya Kundu, Arwa S. Kathiria, Viviana I. Risca, Kristiina Kuningas, Kaur Alasoo, William James Greenleaf, Len A. Pennacchio, et al",
         year=2024, doi="10.1101/2024.12.25.630221", origin="R2-T3"),
    dict(slug="patel-2024-dart-eval",
         title="DART-Eval: A Comprehensive DNA Language Model Evaluation Benchmark on Regulatory DNA",
         authors="Aman Patel, Arpita Singhal, Austin Wang, Anusri Pampari, Maya Kasowski, Anshul Kundaje",
         year=2024, arxiv="2412.05430", origin="R2-T3"),
    dict(slug="peebles-2022-dit-adaln",
         title="Scalable Diffusion Models with Transformers",
         authors="William Peebles, Saining Xie",
         year=2022, arxiv="2212.09748", origin="R2-T1"),
    dict(slug="rafi-2024-dream",
         title="A community effort to optimize sequence-based deep learning models of gene regulation",
         authors="Abdul Muntakim Rafi, Daria Nogina, Dmitry Penzar, Dohoon Lee, Danyeong Lee, Nayeon Kim, Sangyeup Kim, Dohyeon Kim, Yeojin Shin, Il-Youp Kwak, Georgy Meshcheryakov, Andrey Lando, et al",
         year=2024, doi="10.1038/s41587-024-02414-w", origin="R2-T3"),
    dict(slug="rigby-2005-gamlss",
         title="Generalized Additive Models for Location, Scale and Shape",
         authors="R. A. Rigby, D. M. Stasinopoulos",
         year=2005, doi="10.1111/j.1467-9876.2005.00510.x", origin="R2-T2"),
    dict(slug="sasse-2023",
         title="Benchmarking of deep neural networks for predicting personal gene expression from DNA sequence highlights shortcomings",
         authors="Alexander Sasse, Bernard Ng, Anna E. Spiro, Shinya Tasaki, David A. Bennett, Christopher Gaiteri, Philip L. De Jager, Maria Chikina, Sara Mostafavi",
         year=2023, doi="10.1038/s41588-023-01524-6", origin="R2-T3"),
    dict(slug="seitzer-2022-seitzer-betanll",
         title="On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks",
         authors="Maximilian Seitzer, Arash Tavakoli, Dimitrije Antic, Georg Martius",
         year=2022, arxiv="2203.09168", origin="R2-T2"),
    dict(slug="shaw-2018-shaw-relpos",
         title="Self-Attention with Relative Position Representations",
         authors="Peter Shaw, Jakob Uszkoreit, Ashish Vaswani",
         year=2018, arxiv="1803.02155", origin="R2-T1"),
    dict(slug="shazeer-2017-moe",
         title="Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer",
         authors="Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean",
         year=2017, arxiv="1701.06538", origin="R2-T1"),
    dict(slug="spiro-2026-sagenet",
         title="A scalable approach to investigating sequence-to-function predictions from personal genomes",
         authors="Anna E. Spiro, Xinming Tu, Yilun Sheng, Alexander Sasse, Rezwan Hosseini, Maria Chikina, Sara Mostafavi",
         year=2026, doi="10.1038/s41592-026-03124-8", origin="R2-T3"),
    dict(slug="sun-2026-succeed",
         title="Large-scale data-driven pre-trained DNA models enhance performance across diverse genomics tasks",
         authors="Canzhuang Sun, Zhijie He, Shifei Zhang, Kang Xu, Yu Sun, Yuyang Wang, Pengzhen Hu, Xiaochen Bo, Mingzhi Liao, Hao Li, Hebing Chen",
         year=2026, doi="10.1038/s41467-026-73129-6", origin="R2-T3"),
    dict(slug="svensson-2020",
         title="Droplet scRNA-seq is not zero-inflated",
         authors="Valentine Svensson",
         year=2020, doi="10.1038/s41587-019-0379-5", origin="R2-T3"),
    dict(slug="tang-2025-tang-glm",
         title="Evaluating the representational power of pre-trained DNA language models for regulatory genomics",
         authors="Ziqi Tang, Nirali Somia, Yiyang Yu, Peter K. Koo",
         year=2025, doi="10.1186/s13059-025-03674-8", origin="R2-T3"),
    dict(slug="toneyan-2022",
         title="Evaluating deep learning for predicting epigenomic profiles",
         authors="Shushan Toneyan, Ziqi Tang, Peter K. Koo",
         year=2022, doi="10.1038/s42256-022-00570-9", origin="R2-T3"),
    dict(slug="townes-2019-glmpca",
         title="Feature selection and dimension reduction for single-cell RNA-Seq based on a multinomial model",
         authors="F. William Townes, Stephanie C. Hicks, Martin J. Aryee, Rafael A. Irizarry",
         year=2019, doi="10.1186/s13059-019-1861-6", origin="R2-T3"),
    dict(slug="vafa-2025-steerability",
         title="What's Producible May Not Be Reachable: Measuring the Steerability of Generative Models",
         authors="Keyon Vafa, Sarah Bentley, Jon Kleinberg, Sendhil Mullainathan",
         year=2025, arxiv="2503.17482", origin="R2-T2"),
    dict(slug="yang-2019-condconv",
         title="CondConv: Conditionally Parameterized Convolutions for Efficient Inference",
         authors="Brandon Yang, Gabriel Bender, Quoc V. Le, Jiquan Ngiam",
         year=2019, arxiv="1904.04971", origin="R2-T1"),
    dict(slug="young-2024-ddpn",
         title="Fully Heteroscedastic Count Regression with Deep Double Poisson Networks",
         authors="Spencer Young, Porter Jenkins, Longchao Da, Jeff Dotson, Hua Wei",
         year=2024, arxiv="2406.09262", origin="R2-T2"),
    dict(slug="zhang-2025-mpra-eval",
         title="Comprehensive evaluation of diverse massively parallel reporter assays to functionally characterize human enhancers genome-wide",
         authors="Junke Zhang, Alden King-Yung Leung, Yutong Zhu, Li Yao, Avery Willis, Xiuqi Pan, Abdullah Ozer, Zhou Zhou, Keith Siklenka, Alejandro Barrera, Jin Liang, Nathaniel D. Tippens, et al",
         year=2025, doi="10.1186/s13059-025-03828-8", origin="R2-T3"),
    dict(slug="zhou-2026-degu",
         title="Uncertainty-aware genomic deep learning with knowledge distillation",
         authors="Jessica Zhou, Kaeli Rizzo, Trevor Christensen, Ziqi Tang, Peter K. Koo",
         year=2026, doi="10.1038/s44387-025-00053-3", origin="R2-T2"),

    # ---------------- Round 3: the training-objective layer ----------------
    # Added after the 2026-08-01 multi-axis audit found no wiki coverage of class imbalance,
    # multi-task gradient conflict, alternative regression likelihoods, or the optimisers
    # production actually uses. See wiki/imbalance-aware-objectives.md,
    # wiki/multi-task-optimization.md, wiki/regression-likelihoods.md, wiki/training-mechanics.md.
    dict(slug="lin-2017-focal-loss",
         title="Focal Loss for Dense Object Detection",
         authors="Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar",
         year=2017, arxiv="1708.02002", origin="R3"),
    dict(slug="cui-2019-class-balanced-loss",
         title="Class-Balanced Loss Based on Effective Number of Samples",
         authors="Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie",
         year=2019, arxiv="1901.05555", origin="R3"),
    dict(slug="shrivastava-2016-ohem",
         title="Training Region-based Object Detectors with Online Hard Example Mining",
         authors="Abhinav Shrivastava, Abhinav Gupta, Ross Girshick",
         year=2016, arxiv="1604.03540", origin="R3"),
    dict(slug="yu-2020-pcgrad",
         title="Gradient Surgery for Multi-Task Learning",
         authors="Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn",
         year=2020, arxiv="2001.06782", origin="R3"),
    dict(slug="chen-2018-gradnorm",
         title="GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks",
         authors="Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, Andrew Rabinovich",
         year=2018, arxiv="1711.02257", origin="R3"),
    dict(slug="standley-2020-task-grouping",
         title="Which Tasks Should Be Learned Together in Multi-task Learning?",
         authors="Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, Silvio Savarese",
         year=2020, arxiv="1905.07553", origin="R3"),
    dict(slug="barron-2019-robust-loss",
         title="A General and Adaptive Robust Loss Function",
         authors="Jonathan T. Barron", year=2019, arxiv="1701.03077", origin="R3"),
    dict(slug="kingma-2015-adam-adamax",
         title="Adam: A Method for Stochastic Optimization (introduces AdaMax)",
         authors="Diederik P. Kingma, Jimmy Ba", year=2015, arxiv="1412.6980", origin="R3"),
    dict(slug="liu-2025-muon-scalable",
         title="Muon is Scalable for LLM Training",
         authors="Jingyuan Liu, Jianlin Su, Xingcheng Yao, et al (Moonshot AI)",
         year=2025, arxiv="2502.16982", origin="R3"),
    dict(slug="zhang-2020-gradient-clipping",
         title="Why Gradient Clipping Accelerates Training: A Theoretical Justification for Adaptivity",
         authors="Jingzhao Zhang, Tianxing He, Suvrit Sra, Ali Jadbabaie",
         year=2020, arxiv="1905.11881", origin="R3"),
    dict(slug="zhang-2020-heavy-tailed-noise",
         title="Why are Adaptive Methods Good for Attention Models?",
         authors="Jingzhao Zhang, Sai Praneeth Karimireddy, Andreas Veit, Seungyeon Kim, Sashank Reddi, Sanjiv Kumar, Suvrit Sra",
         year=2020, arxiv="1912.03194", origin="R3"),
    dict(slug="detlefsen-2019-variance-networks",
         title="Reliable training and estimation of variance networks",
         authors="Nicki S. Detlefsen, Martin Jorgensen, Soren Hauberg",
         year=2019, arxiv="1906.03260", origin="R3"),
    dict(slug="singh-2016-deepchrome",
         title="DeepChrome: Deep-learning for predicting gene expression from histone modifications",
         authors="Ritambhara Singh, Jack Lanchantin, Gabriel Robins, Yanjun Qi",
         year=2016, arxiv="1607.02078", origin="R3"),
    dict(slug="shrikumar-2017-revcomp-parameter-sharing",
         title="Reverse-complement parameter sharing improves deep learning models for genomics",
         authors="Avanti Shrikumar, Peyton Greenside, Anshul Kundaje",
         year=2017, doi="10.1101/103663", biorxiv=True, origin="R3"),
    dict(slug="saito-2015-precision-recall-plot",
         title="The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets",
         authors="Takaya Saito, Marc Rehmsmeier",
         year=2015, doi="10.1371/journal.pone.0118432", pmcid="PMC4349800", origin="R3"),

]

# Software / web resources from the challenge paper's reference list — no PDF exists.
RESOURCES = [
    ("ENCODE Imputation Challenge (Synapse syn6131484)", "https://www.synapse.org/#!Synapse:syn6131484/wiki/", "C#26"),
    ("ENCODE Imputation Challenge Scoring (Lee JW, 2019)", "https://github.com/ENCODE-DCC/imputation_challenge", "C#27"),
    ("ENCODE-DCC ATAC-seq pipeline (Lee J et al, 2019)", "https://github.com/ENCODE-DCC/atac-seq-pipeline", "C#30"),
    ("ENCODE-DCC ChIP-seq pipeline2 v1.9.0 (Lee J et al, 2021)", "https://github.com/ENCODE-DCC/chip-seq-pipeline2", "C#34"),
]

import os, re, shutil, sys, time, difflib, json
import requests

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw")
EMAIL = "mehdiforoozandehsh@gmail.com"
UA = {"User-Agent": f"crux-wiki-fetch/1.0 (mailto:{EMAIL})"}
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
os.makedirs(RAW, exist_ok=True)

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def similar(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()

def save(path, content, kind="pdf"):
    if kind == "pdf":
        if not content.startswith(b"%PDF"):
            return False
        if len(content) < 15000:
            return False
    else:
        if len(content) < 3000:
            return False
    with open(path, "wb") as f:
        f.write(content)
    return True

def get(url, **kw):
    try:
        r = requests.get(url, headers=UA, timeout=60, allow_redirects=True, **kw)
        if r.status_code == 200:
            return r
    except Exception as e:
        print(f"      ! {type(e).__name__}: {str(e)[:90]}")
    return None

def epmc_lookup(entry):
    """Return (pmcid, doi, title, is_oa) or None."""
    queries = []
    if entry.get("doi"):
        queries.append(f'DOI:"{entry["doi"]}"')
    queries.append(f'TITLE:"{entry["title"]}"')
    for q in queries:
        r = get(f"{EPMC}/search", params={"query": q, "format": "json", "resultType": "core", "pageSize": 5})
        if not r:
            continue
        try:
            results = r.json().get("resultList", {}).get("result", [])
        except Exception:
            continue
        for res in results:
            if similar(res.get("title", ""), entry["title"]) < 0.80:
                continue
            pmcid = res.get("pmcid") or (res.get("fullTextIdList", {}).get("fullTextId") or [None])[0]
            is_oa = res.get("isOpenAccess") == "Y"
            return pmcid, res.get("doi") or entry.get("doi"), res.get("title"), is_oa
        time.sleep(0.4)
    return None

def try_epmc_pdf(pmcid):
    for url in (f"{EPMC}/{pmcid}/fullTextPDF",
                f"https://europepmc.org/api/fulltextRepo?pprId={pmcid}&type=FILE&fileName=EMS.pdf"):
        r = get(url)
        if r and r.content.startswith(b"%PDF"):
            return r.content
    return None

def try_epmc_xml(pmcid):
    r = get(f"{EPMC}/{pmcid}/fullTextXML")
    if r and b"<article" in r.content[:4000]:
        return r.content
    return None

def try_unpaywall(doi):
    r = get(f"https://api.unpaywall.org/v2/{doi}", params={"email": EMAIL})
    if not r:
        return None, None
    try:
        d = r.json()
    except Exception:
        return None, None
    locs = []
    if d.get("best_oa_location"):
        locs.append(d["best_oa_location"])
    locs += [l for l in (d.get("oa_locations") or []) if l not in locs]
    for loc in locs:
        for key in ("url_for_pdf", "url"):
            u = loc.get(key)
            if not u:
                continue
            r2 = get(u)
            if r2 and r2.content.startswith(b"%PDF"):
                return r2.content, "pdf"
            if r2 and key == "url" and b"<html" in r2.content[:2000].lower() and len(r2.content) > 20000:
                return r2.content, "html"
    return None, None

def try_biorxiv(entry):
    r = get("https://api.biorxiv.org/details/biorxiv/" + (entry.get("doi") or ""))
    # fall back to a title search on bioRxiv's public search is unreliable; rely on doi when present
    if r:
        try:
            coll = r.json().get("collection", [])
            if coll:
                doi = coll[-1]["doi"]
                v = coll[-1].get("version", "1")
                r2 = get(f"https://www.biorxiv.org/content/{doi}v{v}.full.pdf")
                if r2 and r2.content.startswith(b"%PDF"):
                    return r2.content
        except Exception:
            pass
    return None

report = []
for i, e in enumerate(ENTRIES, 1):
    slug = e["slug"]
    pdf_path = os.path.join(RAW, slug + ".pdf")
    xml_path = os.path.join(RAW, slug + ".xml")
    html_path = os.path.join(RAW, slug + ".html")
    if any(os.path.exists(p) for p in (pdf_path, xml_path, html_path)):
        got = [p for p in (pdf_path, xml_path, html_path) if os.path.exists(p)][0]
        print(f"[{i:02d}/{len(ENTRIES)}] {slug}: already have {os.path.basename(got)}")
        report.append(dict(slug=slug, status="cached", file=os.path.basename(got)))
        continue

    print(f"[{i:02d}/{len(ENTRIES)}] {slug}")
    status, fname, via = "MISSING", None, None

    # 0. local copy
    if e.get("local") and os.path.exists(e["local"]):
        shutil.copy2(e["local"], pdf_path)
        status, fname, via = "ok", slug + ".pdf", "local"

    # 1. arXiv
    if status != "ok" and e.get("arxiv"):
        r = get(f"https://arxiv.org/pdf/{e['arxiv']}")
        if r and save(pdf_path, r.content):
            status, fname, via = "ok", slug + ".pdf", "arxiv"

    # 2. explicit url
    if status != "ok" and e.get("url"):
        r = get(e["url"])
        if r and save(pdf_path, r.content):
            status, fname, via = "ok", slug + ".pdf", "direct-url"

    # 3. Europe PMC
    doi = e.get("doi")
    if status != "ok":
        look = epmc_lookup(e)
        if look:
            pmcid, doi_found, title_found, is_oa = look
            doi = doi_found or doi
            if pmcid:
                c = try_epmc_pdf(pmcid)
                if c and save(pdf_path, c):
                    status, fname, via = "ok", slug + ".pdf", f"epmc-pdf/{pmcid}"
                else:
                    x = try_epmc_xml(pmcid)
                    if x and save(xml_path, x, kind="xml"):
                        status, fname, via = "ok", slug + ".xml", f"epmc-xml/{pmcid}"
        time.sleep(0.4)

    # 4. Unpaywall
    if status != "ok" and doi:
        c, kind = try_unpaywall(doi)
        if c:
            p = pdf_path if kind == "pdf" else html_path
            if save(p, c, kind=kind):
                status, fname, via = "ok", os.path.basename(p), f"unpaywall-{kind}"
        time.sleep(0.4)

    # 5. bioRxiv
    if status != "ok" and (e.get("biorxiv") or (doi or "").startswith("10.1101/2")):
        c = try_biorxiv(e)
        if c and save(pdf_path, c):
            status, fname, via = "ok", slug + ".pdf", "biorxiv"

    print(f"      -> {status} {fname or ''} {('via ' + via) if via else ''}")
    report.append(dict(slug=slug, status=status, file=fname, via=via,
                       doi=doi, title=e["title"], authors=e["authors"], year=e["year"],
                       origin=e["origin"]))
    time.sleep(0.5)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_report.json")
with open(out, "w") as f:
    json.dump(report, f, indent=2)
ok = sum(1 for r in report if r["status"] in ("ok", "cached"))
print(f"\n=== {ok}/{len(ENTRIES)} retrieved; report -> {out} ===")
for r in report:
    if r["status"] == "MISSING":
        print(f"  MISSING: {r['slug']}  ({r.get('doi')})")
