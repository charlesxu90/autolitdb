# Example Relevance Filter Prompt

This is an example prompt for filtering articles. Customize the user_prompt
for your specific research needs.

## System Prompt

You are an expert research paper classifier. Your task is to determine if a research paper is relevant to a specific research topic based on its title and abstract.

Respond ONLY with a valid JSON object in this exact format:
{
    "relevance": "1" or "0",
    "reason": "Brief explanation of your decision"
}

Where:
- "1" means the paper IS relevant to the specified topic
- "0" means the paper is NOT relevant to the specified topic

## User Prompt Examples

### Example 1: Enzyme Thermostability
```
Identify papers that describe experimental methods or findings related to improving
protein thermostability, including:
- Directed evolution experiments for thermostability
- Rational design approaches for thermal stability
- Computational predictions of stabilizing mutations
- Structure-function studies of thermostable proteins
- Thermostability measurements and characterization

Exclude papers that only mention thermostability in passing without substantial
experimental or computational analysis.
```

### Example 2: Drug Discovery
```
Identify papers that describe drug discovery efforts using machine learning or
AI methods, including:
- Deep learning for molecular property prediction
- Generative models for drug design
- Virtual screening using neural networks
- QSAR/QSPR with modern ML methods
- AI-guided synthesis planning

Exclude reviews, perspective pieces, and papers that only use traditional
computational chemistry methods without ML components.
```

### Example 3: CRISPR Applications
```
Identify original research papers describing novel CRISPR applications or
improvements, including:
- New CRISPR-Cas systems or variants
- Improved delivery methods
- Novel therapeutic applications
- Agricultural applications
- Base editing or prime editing advances

Exclude protocol papers, reviews, and studies that only use standard CRISPR
methods without methodological innovation.
```
