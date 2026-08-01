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
    -> save as raw/guo-2017-tfbs-imputation-matrix-completion.pdf
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
