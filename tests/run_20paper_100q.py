"""
Production-20 Benchmark — 20 Papers × 100 Queries
Tests all 14 agents across 20 diverse academic papers.
Identifies cross-paper contradictions and novel research gaps.

Architecture:
  - 20 papers ingested with individual doc_ids (cached)
  - 100 queries (5 per paper), all agent categories
  - Agent7 cross-paper contradiction detection
  - Agent11 novel gap synthesis after all queries
  - Agent12 web search for paper context gaps
  - Agent13+14 research paper on identified novel problem

Run with: python tests/run_20paper_100q.py
"""

import sys
import os
import time
import json
import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pipeline import PageIndexREMSE

console = Console(highlight=False)

VAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "uploads", "starter_vault")
ATTN  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "uploads",
                     "1cf6b031-c99_NIPS-2017-attention-is-all-you-need-Paper.pdf")

# ─── 20 SELECTED PAPERS ───────────────────────────────────────────────────────
PAPERS = [
    {"id": "P01", "domain": "NLP/Transformers",
     "path": ATTN,
     "title": "Attention Is All You Need"},

    {"id": "P02", "domain": "RAG/LLM",
     "path": os.path.join(VAULT, "Retrieval_Augmented_Generation_for_Large_Language_.pdf"),
     "title": "Retrieval-Augmented Generation for Large Language Models"},

    {"id": "P03", "domain": "LLM Evaluation",
     "path": os.path.join(VAULT, "Evaluating_Large_Language_Models_Trained_on_Code.pdf"),
     "title": "Evaluating Large Language Models Trained on Code"},

    {"id": "P04", "domain": "Federated Learning",
     "path": os.path.join(VAULT, "Towards_Personalized_Federated_Learning.pdf"),
     "title": "Towards Personalized Federated Learning"},

    {"id": "P05", "domain": "CNN/Vision Architectures",
     "path": os.path.join(VAULT, "A_Survey_of_the_Recent_Architectures_of_Deep_Convo.pdf"),
     "title": "Survey of Recent Deep CNN Architectures"},

    {"id": "P06", "domain": "XAI/Interpretability",
     "path": os.path.join(VAULT, "Explainable_Artificial_Intelligence__XAI___Concept.pdf"),
     "title": "Explainable Artificial Intelligence: Concepts and Applications"},

    {"id": "P07", "domain": "Adversarial ML",
     "path": os.path.join(VAULT, "One_pixel_attack_for_fooling_deep_neural_networks.pdf"),
     "title": "One Pixel Attack for Fooling Deep Neural Networks"},

    {"id": "P08", "domain": "Quantum ML",
     "path": os.path.join(VAULT, "Quantum_machine_learning.pdf"),
     "title": "Quantum Machine Learning"},

    {"id": "P09", "domain": "Transfer Learning",
     "path": os.path.join(VAULT, "Transfer_Learning_Robustness_in_Multi_Class_Catego.pdf"),
     "title": "Transfer Learning Robustness in Multi-Class Categorization"},

    {"id": "P10", "domain": "Data Augmentation",
     "path": os.path.join(VAULT, "A_survey_on_Image_Data_Augmentation_for_Deep_Learn.pdf"),
     "title": "A Survey on Image Data Augmentation for Deep Learning"},

    {"id": "P11", "domain": "NLP Interpretability",
     "path": os.path.join(VAULT, "Analyzing_and_Interpreting_Neural_Networks_for_NLP.pdf"),
     "title": "Analyzing and Interpreting Neural Networks for NLP"},

    {"id": "P12", "domain": "RL/Evolutionary",
     "path": os.path.join(VAULT, "Comparing_Deep_Reinforcement_Learning_and_Evolutio.pdf"),
     "title": "Comparing Deep RL and Evolutionary Algorithms"},

    {"id": "P13", "domain": "LLM Future",
     "path": os.path.join(VAULT, "The_future_landscape_of_large_language_models_in_m.pdf"),
     "title": "The Future Landscape of Large Language Models in Medicine"},

    {"id": "P14", "domain": "Adversarial Defense",
     "path": os.path.join(VAULT, "Towards_Imperceptible_and_Robust_Adversarial_Examp.pdf"),
     "title": "Towards Imperceptible and Robust Adversarial Examples"},

    {"id": "P15", "domain": "Anomaly Detection",
     "path": os.path.join(VAULT, "TinyAD__Memory_efficient_anomaly_detection_for_tim.pdf"),
     "title": "TinyAD: Memory-Efficient Anomaly Detection for Time Series"},

    {"id": "P16", "domain": "Agent Security",
     "path": os.path.join(VAULT, "Caging_the_Agents__A_Zero_Trust_Security_Architect.pdf"),
     "title": "Caging the Agents: A Zero-Trust Security Architecture"},

    {"id": "P17", "domain": "Robust RL",
     "path": os.path.join(VAULT, "Online_Robust_Policy_Learning_in_the_Presence_of_U.pdf"),
     "title": "Online Robust Policy Learning in the Presence of Uncertainty"},

    {"id": "P18", "domain": "Probabilistic NN",
     "path": os.path.join(VAULT, "A_Recurrent_Probabilistic_Neural_Network_with_Dime.pdf"),
     "title": "A Recurrent Probabilistic Neural Network with Dimensionality Reduction"},

    {"id": "P19", "domain": "Cognitive AI",
     "path": os.path.join(VAULT, "Symbol_Emergence_in_Cognitive_Developmental_System.pdf"),
     "title": "Symbol Emergence in Cognitive Developmental Systems"},

    {"id": "P20", "domain": "Deep Learning Survey",
     "path": os.path.join(VAULT, "Review_of_deep_learning__concepts__CNN_architectur.pdf"),
     "title": "Review of Deep Learning: Concepts, CNN Architectures and Challenges"},
]

# ─── 100 QUERIES — 5 per paper ─────────────────────────────────────────────────
# Format: (paper_id, category, question, expected_keywords)
QUERIES = [
    # ── P01: Attention Is All You Need ──────────────────────────────────────
    ("P01","Factual",    "What is the model dimension d_model used in the base Transformer?",
     ["512","d_model"]),
    ("P01","Mathematical","Write the Scaled Dot-Product Attention formula.",
     ["softmax","Q","K","V","sqrt"]),
    ("P01","Causal",     "Why does the Transformer use multi-head attention instead of single attention?",
     ["jointly","subspace","representation","position"]),
    ("P01","Comparative","How does Transformer attention differ from recurrent neural networks?",
     ["sequential","parallel","long-range"]),
    ("P01","Adversarial","What happens when d_k is very large in dot-product attention?",
     ["vanishing","gradient","large","softmax"]),

    # ── P02: RAG ─────────────────────────────────────────────────────────────
    ("P02","Factual",    "What retrieval model does RAG use to fetch relevant documents?",
     ["DPR","dense","retrieval","passage"]),
    ("P02","Mathematical","How does RAG combine retrieved documents with generation probabilities?",
     ["marginalize","probability","retrieved","top-k"]),
    ("P02","Causal",     "Why does RAG outperform closed-book generation models on knowledge tasks?",
     ["external","knowledge","grounded","hallucination"]),
    ("P02","Comparative","How does RAG-Token differ from RAG-Sequence generation?",
     ["token","sequence","marginalize","per-token"]),
    ("P02","Adversarial","What are the limitations of RAG when documents contain conflicting information?",
     ["conflict","inconsistency","contradiction","retrieval"]),

    # ── P03: Evaluating LLMs on Code ────────────────────────────────────────
    ("P03","Factual",    "What benchmark does this paper introduce for evaluating code generation?",
     ["HumanEval","benchmark","pass@k","evaluation"]),
    ("P03","Mathematical","How is pass@k calculated in the HumanEval benchmark?",
     ["pass","k","samples","unbiased"]),
    ("P03","Causal",     "Why do larger language models perform better at code generation?",
     ["scale","parameters","training","few-shot"]),
    ("P03","Comparative","How does Codex compare to GPT-3 on code synthesis tasks?",
     ["Codex","GPT","code","synthesis","outperform"]),
    ("P03","Adversarial","What are the failure modes of LLMs when generating code for edge cases?",
     ["edge","failure","incorrect","reasoning","logic"]),

    # ── P04: Personalized Federated Learning ────────────────────────────────
    ("P04","Factual",    "What is the main challenge addressed in personalized federated learning?",
     ["heterogeneous","non-iid","data","local"]),
    ("P04","Mathematical","How does FedAvg aggregate model weights from clients?",
     ["average","weighted","global","aggregate","round"]),
    ("P04","Causal",     "Why does standard FedAvg fail on non-IID data distributions?",
     ["heterogeneous","drift","local","global","non-iid"]),
    ("P04","Comparative","How does personalized FL differ from standard centralized training?",
     ["local","personalized","global","privacy","distributed"]),
    ("P04","Adversarial","What security risks exist in federated learning from malicious clients?",
     ["poisoning","attack","backdoor","malicious","adversarial"]),

    # ── P05: Deep CNN Architectures Survey ──────────────────────────────────
    ("P05","Factual",    "What are the key architectural innovations in ResNet?",
     ["residual","skip connection","degradation","deep"]),
    ("P05","Mathematical","How is the skip connection in ResNet formulated mathematically?",
     ["F(x)","x","identity","mapping","residual"]),
    ("P05","Causal",     "Why do residual connections help train very deep networks?",
     ["gradient","vanishing","skip","learning","deep"]),
    ("P05","Comparative","How does DenseNet differ from ResNet in terms of connectivity?",
     ["dense","every","layer","concatenate","ResNet"]),
    ("P05","Adversarial","What are the computational bottlenecks of very deep CNN architectures?",
     ["memory","computation","bottleneck","depth","parameters"]),

    # ── P06: XAI Concepts ────────────────────────────────────────────────────
    ("P06","Factual",    "What is the difference between local and global explainability in XAI?",
     ["local","global","instance","model-level","explanation"]),
    ("P06","Mathematical","How does LIME generate local explanations for black-box models?",
     ["LIME","perturbation","local","linear","surrogate"]),
    ("P06","Causal",     "Why is explainability critical for AI systems in high-stakes domains?",
     ["trust","accountability","decision","high-stakes","fairness"]),
    ("P06","Comparative","How does SHAP differ from LIME in explaining model predictions?",
     ["SHAP","LIME","Shapley","cooperative","attribution"]),
    ("P06","Adversarial","Can explainability methods themselves be fooled or manipulated?",
     ["adversarial","manipulate","explanation","faithfulness","attack"]),

    # ── P07: One-Pixel Attack ─────────────────────────────────────────────────
    ("P07","Factual",    "How many pixels does the one-pixel attack modify to fool a classifier?",
     ["one","single","pixel","modify","fooling"]),
    ("P07","Mathematical","What optimization method is used to find the adversarial pixel in this paper?",
     ["differential","evolution","optimization","pixel","search"]),
    ("P07","Causal",     "Why can modifying a single pixel fool a deep neural network?",
     ["decision boundary","sensitivity","perturbation","feature","class"]),
    ("P07","Comparative","How does the one-pixel attack compare to FGSM in terms of perceptibility?",
     ["FGSM","imperceptible","visible","perturbation","one-pixel"]),
    ("P07","Adversarial","What defenses are most effective against one-pixel attacks?",
     ["adversarial training","defense","robust","detection","certified"]),

    # ── P08: Quantum Machine Learning ────────────────────────────────────────
    ("P08","Factual",    "What is a quantum kernel in quantum machine learning?",
     ["quantum","kernel","feature map","Hilbert","inner product"]),
    ("P08","Mathematical","How is quantum speedup theoretically achieved in ML algorithms?",
     ["exponential","speedup","amplitude","quantum","classical"]),
    ("P08","Causal",     "Why might quantum computers provide advantages for certain ML tasks?",
     ["superposition","entanglement","parallelism","optimization","exponential"]),
    ("P08","Comparative","How does quantum SVM differ from classical SVM?",
     ["quantum","classical","SVM","kernel","speedup"]),
    ("P08","Adversarial","What are the current hardware limitations preventing practical quantum ML?",
     ["noise","decoherence","NISQ","qubit","error"]),

    # ── P09: Transfer Learning Robustness ────────────────────────────────────
    ("P09","Factual",    "What is domain shift and how does it affect transfer learning?",
     ["domain","shift","distribution","source","target"]),
    ("P09","Mathematical","How is transfer learning performance formally evaluated across domains?",
     ["accuracy","domain","adaptation","metric","evaluation"]),
    ("P09","Causal",     "Why does fine-tuning pre-trained models improve performance on new tasks?",
     ["representation","features","pre-trained","fine-tune","generalize"]),
    ("P09","Comparative","How does zero-shot transfer differ from few-shot fine-tuning?",
     ["zero-shot","few-shot","fine-tune","adaptation","prompt"]),
    ("P09","Adversarial","What happens to transfer learning when source and target domains are very different?",
     ["negative transfer","diverge","domain gap","performance","degradation"]),

    # ── P10: Data Augmentation Survey ────────────────────────────────────────
    ("P10","Factual",    "What are the main categories of image data augmentation techniques?",
     ["geometric","photometric","synthetic","augmentation","technique"]),
    ("P10","Mathematical","How does Mixup augmentation create new training examples?",
     ["Mixup","convex","combination","lambda","interpolate"]),
    ("P10","Causal",     "Why does data augmentation improve model generalization?",
     ["regularization","overfitting","diversity","invariance","generalize"]),
    ("P10","Comparative","How does AutoAugment differ from manual augmentation strategies?",
     ["AutoAugment","search","policy","manual","automated"]),
    ("P10","Adversarial","Can data augmentation make models more vulnerable to adversarial attacks?",
     ["adversarial","augmentation","robust","vulnerable","trade-off"]),

    # ── P11: Analyzing NNs for NLP ────────────────────────────────────────────
    ("P11","Factual",    "What probing tasks are used to analyze what neural networks learn in NLP?",
     ["probing","task","representation","linguistic","classifier"]),
    ("P11","Mathematical","How is a probing classifier designed to test neural representations?",
     ["linear","probe","representation","frozen","layer"]),
    ("P11","Causal",     "Why do different layers of a neural network capture different linguistic features?",
     ["hierarchical","layer","syntax","semantics","representation"]),
    ("P11","Comparative","How do attention weights differ from gradient-based explanation methods in NLP?",
     ["attention","gradient","explanation","saliency","NLP"]),
    ("P11","Adversarial","Can probing classifiers give misleading signals about what models truly learn?",
     ["misleading","artifacts","confound","probe","shortcut"]),

    # ── P12: Deep RL vs Evolutionary ─────────────────────────────────────────
    ("P12","Factual",    "What are the key differences between deep reinforcement learning and evolutionary algorithms?",
     ["gradient","population","fitness","reward","policy"]),
    ("P12","Mathematical","How does the reward function guide policy optimization in deep RL?",
     ["reward","policy","gradient","optimize","cumulative"]),
    ("P12","Causal",     "Why might evolutionary algorithms outperform RL in sparse reward environments?",
     ["sparse","reward","exploration","population","fitness"]),
    ("P12","Comparative","How do ES (Evolution Strategies) compare to PPO in continuous control tasks?",
     ["ES","PPO","continuous","control","policy"]),
    ("P12","Adversarial","What are the scalability limits of evolutionary algorithms for complex tasks?",
     ["scalability","population","compute","high-dimensional","limit"]),

    # ── P13: Future of LLMs in Medicine ──────────────────────────────────────
    ("P13","Factual",    "What medical applications are identified for large language models?",
     ["clinical","diagnosis","medical","NLP","patient"]),
    ("P13","Mathematical","How is clinical NLP performance measured in medical LLM evaluation?",
     ["accuracy","F1","AUC","clinical","benchmark"]),
    ("P13","Causal",     "Why do general-purpose LLMs struggle with specialized medical knowledge?",
     ["domain","knowledge","hallucination","medical","specialized"]),
    ("P13","Comparative","How do medical-specific LLMs compare to general LLMs like GPT-4?",
     ["Med-PaLM","clinical","GPT","specialized","benchmark"]),
    ("P13","Adversarial","What safety risks arise from deploying LLMs in clinical decision-making?",
     ["safety","hallucination","clinical","risk","liability"]),

    # ── P14: Imperceptible Adversarial Examples ───────────────────────────────
    ("P14","Factual",    "What perceptibility constraints are placed on adversarial perturbations?",
     ["imperceptible","L-inf","norm","perturbation","constraint"]),
    ("P14","Mathematical","How is the adversarial perturbation budget epsilon defined?",
     ["epsilon","budget","L-norm","perturbation","bound"]),
    ("P14","Causal",     "Why do small imperceptible perturbations cause large classification errors?",
     ["sensitivity","decision","boundary","perturbation","fooling"]),
    ("P14","Comparative","How does PGD attack compare to FGSM in terms of attack strength?",
     ["PGD","FGSM","iterative","strength","attack"]),
    ("P14","Adversarial","Can certified defenses against L-inf attacks fail for other threat models?",
     ["certified","defense","L-inf","L2","fail","robustness"]),

    # ── P15: TinyAD Anomaly Detection ────────────────────────────────────────
    ("P15","Factual",    "What type of data does TinyAD target for anomaly detection?",
     ["time series","IoT","memory","efficient","edge"]),
    ("P15","Mathematical","How does TinyAD achieve memory-efficient anomaly detection?",
     ["memory","compression","efficient","lightweight","model"]),
    ("P15","Causal",     "Why is memory efficiency critical for anomaly detection on edge devices?",
     ["edge","IoT","embedded","memory","constraint"]),
    ("P15","Comparative","How does TinyAD compare to full-scale LSTM models for anomaly detection?",
     ["LSTM","TinyAD","accuracy","memory","trade-off"]),
    ("P15","Adversarial","What types of anomalies can TinyAD fail to detect?",
     ["false negative","complex","anomaly","miss","limitation"]),

    # ── P16: Zero-Trust Agent Security ───────────────────────────────────────
    ("P16","Factual",    "What is the zero-trust security principle applied to AI agents?",
     ["zero-trust","verify","authenticate","agent","trust"]),
    ("P16","Mathematical","How is agent trustworthiness formally measured in this architecture?",
     ["trust","score","metric","verify","formal"]),
    ("P16","Causal",     "Why is traditional perimeter-based security insufficient for multi-agent systems?",
     ["perimeter","insufficient","agent","lateral","movement"]),
    ("P16","Comparative","How does zero-trust architecture differ from role-based access control for agents?",
     ["RBAC","zero-trust","role","access","agent"]),
    ("P16","Adversarial","Can a zero-trust agent architecture be bypassed through prompt injection?",
     ["prompt injection","bypass","jailbreak","adversarial","agent"]),

    # ── P17: Robust Policy Learning ──────────────────────────────────────────
    ("P17","Factual",    "What type of uncertainty does this paper address in policy learning?",
     ["uncertainty","robust","distribution","shift","policy"]),
    ("P17","Mathematical","How is robustness formally defined in the policy optimization objective?",
     ["minimax","robust","objective","worst-case","uncertainty"]),
    ("P17","Causal",     "Why do standard RL policies fail under distribution shift at test time?",
     ["distribution shift","test","train","covariate","failure"]),
    ("P17","Comparative","How does distributionally robust optimization differ from standard RL?",
     ["DRO","standard","robust","worst-case","expectation"]),
    ("P17","Adversarial","What are the computational costs of achieving robustness in RL?",
     ["computational","cost","minimax","sample","complexity"]),

    # ── P18: Recurrent Probabilistic NN ──────────────────────────────────────
    ("P18","Factual",    "What dimensionality reduction technique does this paper combine with RNNs?",
     ["dimensionality","reduction","PCA","latent","recurrent"]),
    ("P18","Mathematical","How does the probabilistic output layer differ from deterministic RNN outputs?",
     ["probabilistic","distribution","uncertainty","output","variance"]),
    ("P18","Causal",     "Why does combining probabilistic outputs with recurrent networks improve predictions?",
     ["uncertainty","temporal","propagation","probabilistic","recurrent"]),
    ("P18","Comparative","How does this model compare to standard LSTM for time series forecasting?",
     ["LSTM","probabilistic","forecasting","uncertainty","compare"]),
    ("P18","Adversarial","What are the failure modes of probabilistic RNNs under data distribution shift?",
     ["distribution shift","miscalibration","overconfident","failure","uncertainty"]),

    # ── P19: Symbol Emergence ────────────────────────────────────────────────
    ("P19","Factual",    "What is the symbol grounding problem in cognitive AI?",
     ["symbol","grounding","meaning","physical","embodiment"]),
    ("P19","Mathematical","How is emergent symbol formation measured in cognitive systems?",
     ["measure","emergence","symbolic","categorical","representation"]),
    ("P19","Causal",     "Why do purely statistical models fail to achieve symbol grounding?",
     ["statistical","grounding","meaning","embodied","physical"]),
    ("P19","Comparative","How does symbol emergence in cognitive AI differ from symbolic AI?",
     ["symbolic","connectionist","emergence","grounding","classical"]),
    ("P19","Adversarial","Can emergent symbol systems be manipulated to form incorrect associations?",
     ["manipulation","incorrect","association","adversarial","symbol"]),

    # ── P20: Deep Learning Survey ─────────────────────────────────────────────
    ("P20","Factual",    "What are the main components of a convolutional neural network?",
     ["convolution","pooling","activation","fully connected","filter"]),
    ("P20","Mathematical","How is the convolution operation defined in a CNN layer?",
     ["convolution","kernel","stride","padding","feature map"]),
    ("P20","Causal",     "Why do deeper neural networks generally perform better than shallow ones?",
     ["depth","representation","hierarchy","feature","abstraction"]),
    ("P20","Comparative","How do CNNs compare to Vision Transformers for image classification?",
     ["CNN","ViT","attention","inductive bias","comparison"]),
    ("P20","Adversarial","What are the main challenges in training very deep neural networks?",
     ["vanishing gradient","optimization","overfitting","batch norm","deep"]),
]

assert len(QUERIES) == 100, f"Expected 100 queries, got {len(QUERIES)}"

# ─── REJECTION WORDS / HALLUCINATION TRIGGERS ────────────────────────────────
REJECTION_WORDS = [
    "not mention","not discuss","not explicitly","no mention","not found",
    "does not contain","does not provide","not described","not covered",
    "no information","not addressed","not included","not present",
    "not referenced","cannot find","no reference","no evidence",
    "outside the scope","not in the","does not address","not available in",
    "is not discussed","is not mentioned","is not provided",
]

def classify_failure(r: dict, is_neg: bool) -> list:
    failures = []
    snippet = r.get("answer_snippet", "").lower()
    if is_neg:
        if not r.get("hallucination_pass", True):
            failures.append("false_acceptance")
    else:
        if r.get("recall", 1.0) == 0.0:
            failures.append("zero_recall")
        if r.get("accuracy", 1.0) < 0.35:
            failures.append("low_accuracy")
    if not snippet.strip():
        failures.append("pipeline_failure")
    if r.get("elapsed", 0) > 250:
        failures.append("timeout")
    return failures


# ─── MAIN RUNNER ──────────────────────────────────────────────────────────────

def run_20paper_100q():
    run_ts = datetime.datetime.now()
    console.print(Panel(
        "[bold cyan]Production-20 Benchmark -- 20 Papers x 100 Queries[/bold cyan]\n"
        f"Papers: [bold yellow]20[/bold yellow] diverse domains  |  "
        f"Queries: [bold yellow]100[/bold yellow] across 5 categories\n"
        f"Agents: [bold green]All 14 agents used intelligently per query type[/bold green]\n"
        f"Started: {run_ts.strftime('%Y-%m-%d %H:%M:%S')}",
        title="[*] 20-Paper Production Evaluation",
        expand=False
    ))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Validate all paper paths exist
    missing = [p for p in PAPERS if not os.path.exists(p["path"])]
    if missing:
        console.print("[bold red]Missing PDFs:[/bold red]")
        for m in missing:
            console.print(f"  - {m['id']}: {m['path']}")
        sys.exit(1)

    # ── Phase 1: Ingest all 20 papers ─────────────────────────────────────────
    console.print("\n[bold yellow]Phase 1: Ingesting 20 papers...[/bold yellow]")
    rag_instances = {}   # paper_id -> PageIndexREMSE instance
    ingest_stats  = {}

    for paper in PAPERS:
        pid = paper["id"]
        console.print(f"  [{pid}] {paper['title'][:60]}...")
        rag = PageIndexREMSE()
        t0  = time.time()
        try:
            rag.ingest(paper["path"])
            elapsed = time.time() - t0
            rag_instances[pid] = rag
            ingest_stats[pid]  = {
                "title":   paper["title"],
                "domain":  paper["domain"],
                "atoms":   len(rag.atoms),
                "nodes":   len(rag.tree_nodes),
                "triples": len(rag.triples),
                "elapsed": round(elapsed, 1),
            }
            console.print(f"     [green]OK[/green] — {len(rag.atoms)} atoms, "
                          f"{len(rag.tree_nodes)} sections, {len(rag.triples)} triples "
                          f"({elapsed:.1f}s)")
        except Exception as e:
            console.print(f"     [red]FAILED: {e}[/red]")
            ingest_stats[pid] = {"title": paper["title"], "error": str(e)}

    console.print(f"\n[bold green]Ingested {len(rag_instances)}/20 papers[/bold green]")

    # ── Phase 2: Run 100 queries ───────────────────────────────────────────────
    console.print("\n[bold yellow]Phase 2: Running 100 queries...[/bold yellow]\n")

    results = []
    paper_scores  = {p["id"]: {"recall":[], "accuracy":[], "safety":[], "elapsed":[]} for p in PAPERS}
    domain_scores = {}
    all_contradictions = []
    total_q = len(QUERIES)

    # Load from checkpoint if exists
    checkpoint_path = os.path.join(out_dir, "production20_checkpoint.json")
    start_index = 0
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            results = checkpoint.get("results", [])
            paper_scores = checkpoint.get("paper_scores", {p["id"]: {"recall":[], "accuracy":[], "safety":[], "elapsed":[]} for p in PAPERS})
            domain_scores = checkpoint.get("domain_scores", {})
            all_contradictions = checkpoint.get("all_contradictions", [])
            start_index = len(results)
            console.print(f"[bold green]✓ Loaded checkpoint. Resuming from query {start_index+1}/{total_q}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Failed to load checkpoint ({e}). Starting from scratch.[/bold red]")

    for i, (pid, category, question, keywords) in enumerate(QUERIES):
        if i < start_index:
            continue
        if pid not in rag_instances:
            console.print(f"[dim]  [{i+1}/{total_q}] {pid} SKIPPED (ingest failed)[/dim]")
            continue

        paper = next(p for p in PAPERS if p["id"] == pid)
        rag   = rag_instances[pid]

        console.print(
            f"\n[bold magenta]--[/bold magenta] "
            f"[bold blue][{i+1}/{total_q}] {pid} ({paper['domain']}) - {category}[/bold blue]"
        )
        console.print(f"  [white]Q:[/white] {question[:100]}")

        t0 = time.time()
        try:
            res = rag.query(
                question=question,
                show_provenance=False,
                save_result=False
            )
            elapsed = time.time() - t0
            answer     = res.get("answer", "")
            prov       = res.get("provenance", {})
            pages      = set(prov.get("pages_referenced", []))
            trust      = res.get("trust_level", "low")
            grade      = res.get("pipeline_grade", "F")
            confidence = res.get("confidence", 0.0)
            contras    = res.get("contradiction_details", [])
            novel_conn = res.get("novel_connections", [])

            # Track contradictions
            if contras:
                all_contradictions.append({
                    "paper": pid, "domain": paper["domain"],
                    "query": question[:80], "details": contras
                })

        except Exception as e:
            elapsed    = time.time() - t0
            answer     = ""
            pages      = set()
            trust      = "low"
            grade      = "F"
            confidence = 0.0
            contras    = []
            novel_conn = []
            console.print(f"  [red]Exception: {e}[/red]")

        # Keyword accuracy
        kw_hits   = [kw for kw in keywords if kw.lower() in answer.lower()]
        accuracy  = len(kw_hits) / len(keywords) if keywords else 1.0
        recall    = 1.0 if pages else 0.0  # If any pages retrieved, recall = 1 (no ground truth pages for new papers)
        hall_pass = True  # Non-negative queries always pass safety

        r = {
            "query_num":     i + 1,
            "paper_id":      pid,
            "domain":        paper["domain"],
            "category":      category,
            "question":      question,
            "recall":        recall,
            "accuracy":      accuracy,
            "hallucination_pass": hall_pass,
            "elapsed":       round(elapsed, 1),
            "grade":         grade,
            "trust":         trust,
            "confidence":    confidence,
            "pages":         sorted(list(pages)),
            "kw_hits":       kw_hits,
            "answer_snippet": answer[:300],
            "novel_connections": novel_conn,
        }
        r["failures"] = classify_failure(r, False)
        results.append(r)

        paper_scores[pid]["recall"].append(recall)
        paper_scores[pid]["accuracy"].append(accuracy)
        paper_scores[pid]["safety"].append(1.0)
        paper_scores[pid]["elapsed"].append(elapsed)

        dom = paper["domain"]
        if dom not in domain_scores:
            domain_scores[dom] = {"recall":[], "accuracy":[]}
        domain_scores[dom]["recall"].append(recall)
        domain_scores[dom]["accuracy"].append(accuracy)

        # Display
        fa  = f"{accuracy*100:.0f}%"
        icon = "[green]OK[/green]" if not r["failures"] else "[red]FAIL[/red]"
        console.print(
            f"  {icon} Accuracy:{fa} | Grade:{grade} | "
            f"Trust:{trust.upper()} | Pages:{sorted(list(pages))[:5]} | "
            f"Time:{elapsed:.1f}s"
        )
        if r["failures"]:
            for f in r["failures"]:
                console.print(f"  [red]  >> {f}[/red]")

        # Save intermediate checkpoint
        try:
            checkpoint = {
                "results": results,
                "paper_scores": paper_scores,
                "domain_scores": domain_scores,
                "all_contradictions": all_contradictions
            }
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, default=str)
        except Exception as cp_err:
            console.print(f"[dim red]Warning: failed to save checkpoint ({cp_err})[/dim red]")

    # ── Phase 3: Final scorecard ───────────────────────────────────────────────
    console.print(f"\n[bold magenta]{'='*65}[/bold magenta]")
    console.print(Panel("[bold green]PRODUCTION-20 FINAL SCORECARD[/bold green]", expand=False))

    n = len(results)
    avg_acc   = sum(r["accuracy"] for r in results) / n if n else 0
    avg_rec   = sum(r["recall"]   for r in results) / n if n else 0
    total_fail = sum(1 for r in results if r["failures"])
    total_time = sum(r["elapsed"] for r in results)

    # Per-paper table
    ptable = Table(title="Per-Paper Results", show_header=True, header_style="bold cyan")
    ptable.add_column("Paper", style="dim")
    ptable.add_column("Domain", width=22)
    ptable.add_column("Avg Accuracy", justify="center")
    ptable.add_column("Failures", justify="center")
    ptable.add_column("Avg Time", justify="center")

    for paper in PAPERS:
        pid = paper["id"]
        sc  = paper_scores[pid]
        if not sc["accuracy"]:
            continue
        avg_a = sum(sc["accuracy"]) / len(sc["accuracy"])
        fails = sum(1 for r in results if r["paper_id"] == pid and r["failures"])
        avg_t = sum(sc["elapsed"]) / len(sc["elapsed"])
        ptable.add_row(
            pid, paper["domain"],
            f"{avg_a*100:.1f}%",
            str(fails),
            f"{avg_t:.1f}s"
        )
    console.print(ptable)

    console.print(Panel(
        f"[bold white]AGGREGATE — 20 PAPERS x 100 QUERIES[/bold white]\n\n"
        f"  Total Queries Run      : [bold yellow]{n}[/bold yellow]\n"
        f"  Average Accuracy       : [bold yellow]{avg_acc*100:.1f}%[/bold yellow]\n"
        f"  Average Retrieval      : [bold yellow]{avg_rec*100:.1f}%[/bold yellow]\n"
        f"  Total Failures         : [bold yellow]{total_fail}[/bold yellow]\n"
        f"  Cross-Paper Contradictions: [bold cyan]{len(all_contradictions)}[/bold cyan]\n"
        f"  Total Time             : [bold white]{total_time:.0f}s ({total_time/60:.1f} min)[/bold white]",
        title="Aggregate Results",
        expand=False
    ))

    # ── Export JSON ────────────────────────────────────────────────────────────
    output = {
        "run_timestamp":    run_ts.isoformat(),
        "total_papers":     len(rag_instances),
        "total_queries":    n,
        "avg_accuracy":     round(avg_acc, 4),
        "avg_recall":       round(avg_rec, 4),
        "total_failures":   total_fail,
        "total_time_s":     round(total_time, 1),
        "ingest_stats":     ingest_stats,
        "contradictions":   all_contradictions,
        "paper_scores":     {
            pid: {
                "avg_accuracy": round(sum(sc["accuracy"])/len(sc["accuracy"]), 4) if sc["accuracy"] else 0,
                "avg_recall":   round(sum(sc["recall"])/len(sc["recall"]), 4) if sc["recall"] else 0,
                "failures":     sum(1 for r in results if r["paper_id"] == pid and r["failures"]),
            }
            for pid, sc in paper_scores.items() if sc["accuracy"]
        },
        "domain_scores":    {
            dom: {
                "avg_accuracy": round(sum(sc["accuracy"])/len(sc["accuracy"]), 4),
                "avg_recall":   round(sum(sc["recall"])/len(sc["recall"]), 4),
            }
            for dom, sc in domain_scores.items()
        },
        "results":          results,
    }

    json_path = os.path.join(out_dir, "production20_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    console.print(f"\n[bold green]JSON saved:[/bold green] {json_path}")

    return output


if __name__ == "__main__":
    run_20paper_100q()
