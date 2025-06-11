class PromptTemplates:
    """Centralized prompt templates for multi-hop RAG systems"""
    REMOVE_SOURCE_PROMPT ="""Okay, I see. The original queries are often more complex and structured as a single, sometimes lengthy, question that might implicitly involve comparisons between what different sources say or how a situation evolved according to different reports. The key is to remove the source attribution smoothly while maintaining this complex single-question structure.

Here's the revised prompt for removing sources, with new examples reflecting this style:

---

You are an expert at query reformulation. Your task is to remove any explicit mentions of information sources from the input 'Original Query' while preserving the core, often complex, single question being asked. The reformulated query should flow naturally.

Requirements:
1.  Your output must be a single section: `### Reformulated Query:`.
2.  Follow the formatting of the examples provided for this section.
3.  The `Reformulated Query` should be grammatically correct and retain the full meaning of the original query, excluding only the source attributions. The removal should be seamless, maintaining the integrity of the single, often comparative or analytical, question.

Instruction:
1.  Carefully analyze the 'Original Query' to identify all phrases that attribute information to a specific source (e.g., "according to 'Source Name'", "as reported by 'Source Name'", "'Source Name' states that", "the 'Source Name' article on X").
2.  Remove these source attribution phrases.
3.  Ensure the remaining query is a coherent and complete single question. Adjust punctuation or minor connecting words as needed for natural language flow. The core analytical or comparative nature of the original question must remain unchanged.

Example 1:
### Original Query: Considering the analysis in "Climate Forward Journal" from March 2023 on glacial retreat in the Himalayas and the subsequent data presented by "Mountain Research Bulletin" in September 2023 for the same region, was there an acceleration in the rate of ice loss noted between the two periods?
### Reformulated Query: Considering an analysis from March 2023 on glacial retreat in the Himalayas and subsequent data presented in September 2023 for the same region, was there an acceleration in the rate of ice loss noted between the two periods?

Example 2:
### Original Query: Does the assessment of "Global Auto Review" regarding the market penetration of electric vehicles in Europe by 2025 differ significantly from the forecast provided by "EV Insights Quarterly" for the same period and region?
### Reformulated Query: Does an assessment regarding the market penetration of electric vehicles in Europe by 2025 differ significantly from another forecast for the same period and region?

Example 3:
### Original Query: What specific technological advancement in battery storage was highlighted by "EnergyTech Today" as a game-changer in early 2022, and how was its potential impact on renewable energy integration described by "Sustainable Futures Magazine" later that year?
### Reformulated Query: What specific technological advancement in battery storage was highlighted as a game-changer in early 2022, and how was its potential impact on renewable energy integration described later that year?
*(Self-correction: My previous attempt for a similar structure was too simplistic. This example better reflects maintaining the flow of a single question with two linked parts, even after source removal.)*

Example 4:
### Original Query: Given the report by "PharmaInnovate News" on the initial trial results of 'Drug Z' for Alzheimer's in Q1, and a follow-up study detailed in "Neurology Advances Digest" in Q3 focusing on side effects, did the overall risk-benefit profile of 'Drug Z' appear to change significantly over these reporting periods?
### Reformulated Query: Given initial trial results of 'Drug Z' for Alzheimer's in Q1, and a follow-up study in Q3 focusing on side effects, did the overall risk-benefit profile of 'Drug Z' appear to change significantly over these reporting periods?

Now, reformulate the following query:
### Original Query: {query}
### Reformulated Query:"""

    IR_COT_REASONING_PROMPT = """You are an expert at multi-step reasoning and fact extraction.
You are given the 'Original Question', the 'Current Fact' (which is a list of facts established in previous steps, if any), a 'Current Small Question' that was just executed, and the 'Relevant Documents for Small Question'.
You are excellent at reasoning based on the 'Current Fact' and analyzing the 'Relevant Documents for Small Question' to deduce a 'New Fact' that contributes to answering the 'Original Question'.
Your primary goal is to extract as much relevant information as possible from the provided documents in each step to build a detailed and comprehensive answer.

Requirements:
1.  Your output must consist of two sections: `### Reasoning:` and `### New Fact:`.
2.  Follow the formatting of the examples provided for these sections.
3.  The `New Fact` should be a comprehensive statement. It must extract **all relevant details** from the 'Relevant Documents for Small Question' that address the 'Current Small Question' and contribute to the 'Original Question'. Prioritize capturing details like names, dates, quantities, and specific claims, rather than being overly brief.

Instruction:
1.  Carefully analyze the 'Original Question' and the 'Current Fact' (a list of previously established facts) to understand the overall goal and what has been established so far. If 'Current Fact' is empty, this is likely the first step.
2.  Examine the 'Current Small Question' and the 'Relevant Documents for Small Question' to identify the specific information sought and found in the current step.
3.  Based on your analysis, write down your step-by-step `Reasoning` process. This should explain how the retrieved information, in all its detail, helps in progressing towards the 'Original Question', considering all entries in 'Current Fact'.
4.  Finally, formulate the `New Fact`. This fact should be as detailed as possible, incorporating all pertinent information from the 'Relevant Documents for Small Question' as identified in your reasoning. This new fact will be added to the list of facts for subsequent steps.
5.  **If the `Relevant Documents for Small Question` do not provide a complete answer to the `Current Small Question` (e.g., they provide the main information but are missing a specific source or date that was asked for), you should still extract the substantial facts that are present. Only output `No new fact` if the documents provide no relevant information at all for the small question.**
6.  If `Relevant Documents for Small Question` answer the `Current Small Question` but does not mention the source and datetime. The `New Fact` MUST directly answer the `Current Small Question` like `Current Small Question` do not ask about source and datetime.
7.  If `Relevant Documents for Small Question` answer the `Current Small Question` cannot answer the `Original Question`, but contain information support the `Original Question`, you should output that into `New Fact`.
8.  Avoid answer "the provided document do not contain information ...", instead answer `No new fact`.

Example 1:
### Original Question: According to reports from "SpaceNews" and "Planetary Society Journal," what was the primary objective of the Artemis I mission, and did it involve a crewed lunar landing?

### Current Fact:
(empty)

### Current Small Question: What does "SpaceNews" report as the primary objective of the Artemis I mission and whether it was crewed?

### Relevant Documents for Small Question:
[Document 1 (SpaceNews): "The Artemis I mission, launched successfully last November, served as an uncrewed flight test of NASA's Space Launch System (SLS) rocket and Orion spacecraft. The primary objective was to demonstrate Orion's capabilities in a lunar orbital environment and ensure a safe re-entry, splashdown, and recovery prior to the first flight with astronauts."]

### Reasoning:
The Original Question asks about the primary objective of Artemis I and its crew status, according to "SpaceNews" and another source. This is the first step. The Current Small Question focuses on the "SpaceNews" report. Document 1 provides several key details: the mission was an "uncrewed flight test," it tested specific hardware ("NASA's Space Launch System (SLS) rocket and Orion spacecraft"), its objective was to "demonstrate Orion's capabilities" and ensure "safe re-entry, splashdown, and recovery," and this was a precursor to future astronaut flights. It also mentions the launch timing ("last November"). To build a comprehensive answer, the New Fact should capture all these details, not just a brief summary.

### New Fact: According to SpaceNews, the Artemis I mission, launched in November, was an uncrewed flight test of NASA's Space Launch System (SLS) and Orion spacecraft. Its primary objective was to demonstrate Orion's capabilities in a lunar orbital environment and to ensure its safe re-entry, splashdown, and recovery before the first crewed flight.

Example 2:
### Original Question: Based on articles from "Climate Change Today" detailing impacts in Europe and "Global Weather Patterns Review" discussing shifts in North America, is there a consensus on whether rising global temperatures are leading to more frequent heatwaves in both regions?

### Current Fact:
- "Climate Change Today" (Oct 2023) states that Europe has experienced a significant increase in the frequency and intensity of heatwaves over the past decade, directly linked by scientists to rising global temperatures.

### Current Small Question: What does "Global Weather Patterns Review" say about the frequency of heatwaves in North America in relation to global temperatures?

### Relevant Documents for Small Question:
[Document 1 (Global Weather Patterns Review): "Recent analysis in North America shows a clear trend: as average global temperatures climb, the continent is witnessing more prolonged and severe heatwave events, particularly in the southwestern United States and parts of Canada."]

### Reasoning:
The Original Question seeks to establish a consensus between two sources on heatwaves in Europe and North America. The 'Current Fact' already covers Europe. The Current Small Question now addresses North America using "Global Weather Patterns Review." Document 1 establishes a clear link between rising global temperatures and heatwaves in North America. To provide a detailed basis for comparison, I need to extract not just the link, but also the characteristics of these events ("prolonged and severe") and the specific regions affected ("southwestern United States and parts of Canada"). This level of detail is crucial for a robust answer to the Original Question.

### New Fact: "Global Weather Patterns Review" reports a clear trend where rising average global temperatures are linked to more prolonged and severe heatwave events in North America, with specific impacts noted in the southwestern United States and parts of Canada.

Example 3:
### Original Question: Comparing the launch strategies of "Starlink" as reported by "Satellite Weekly" and "OneWeb" as detailed in "Constellation Monthly", which service aimed for initial global coverage more rapidly?

### Current Fact:
- "Satellite Weekly" (Jan 2022) reported that Starlink's strategy involves rapidly deploying thousands of satellites into low Earth orbit to achieve near-global coverage within a few years of initial launches.

### Current Small Question: What was OneWeb's initial deployment strategy for achieving coverage, according to "Constellation Monthly"?

### Relevant Documents for Small Question:
[Document 1 (Constellation Monthly): "OneWeb's initial deployment focused on providing coverage to regions above 60 degrees North latitude, including Alaska, Canada, and Northern Europe, before gradually expanding to other parts of the globe."]

### Reasoning:
The Original Question is a comparative one about the speed of achieving global coverage for Starlink vs. OneWeb. The 'Current Fact' establishes Starlink's strategy as rapid and near-global. The Current Small Question looks for OneWeb's strategy. Document 1 describes a fundamentally different, geographically-focused strategy for OneWeb. To create a strong point of comparison, I must extract all the details: the initial focus was on a specific latitude ("above 60 degrees North"), the specific examples of these regions ("Alaska, Canada, and Northern Europe"), and the nature of the plan ("before gradually expanding"). These details are essential to contrast with Starlink's approach.

### New Fact: "Constellation Monthly" reports that OneWeb's initial deployment strategy was not for immediate global coverage, but instead focused on providing service to regions above 60 degrees North latitude, specifically mentioning Alaska, Canada, and Northern Europe, before planning a gradual expansion to the rest of the world.

**Example 4: (Extracting partial but critical information)**
### Original Question: What were the Q4 2023 revenue figures for 'Global Petro' and 'Energy United' as reported by Reuters and Bloomberg?

### Current Fact:
- A report from Reuters indicates that 'Global Petro' posted Q4 2023 revenues of $5.2 billion.

### Current Small Question: What were the Q4 2023 revenue figures for 'Energy United' according to Bloomberg?

### Relevant Documents for Small Question:
[Document 1: "A recent financial filing shows Energy United's revenue for the fourth quarter of 2023 was $4.8 billion, a slight decrease from the previous year. The company cited market volatility as the primary reason for the dip."]

### Reasoning:
The Original Question asks for Q4 2023 revenue for two companies from two specific sources. I have the figure for 'Global Petro' from Reuters. The Current Small Question specifically asks for 'Energy United's' revenue *according to Bloomberg*. The provided document gives a clear revenue figure for Energy United for Q4 2023 ($4.8 billion) and even context (a slight decrease due to market volatility). However, it attributes this to a "financial filing," not to "Bloomberg." According to Instruction #5, instead of returning 'No new fact' because the source doesn't match, I should extract the highly relevant core information. The revenue figure itself is critical to answering the Original Question.

### New Fact: A financial filing states that Energy United's revenue for the fourth quarter of 2023 was $4.8 billion.

**Example 5: (Extracting core data when date is missing)**
### Original Question: What was the main policy change announced by the Federal Transport Agency in their press release on May 15, 2024, regarding electric vehicle subsidies?

### Current Fact:
(empty)

### Current Small Question: What was the main policy change regarding electric vehicle subsidies announced by the Federal Transport Agency on May 15, 2024?

### Relevant Documents for Small Question:
[Document 1 (Federal Government News Portal): "The Federal Transport Agency has updated its subsidy program for electric vehicles. Effective immediately, the maximum available tax credit for new EV purchases will be reduced from $7,500 to $5,000."]

### Reasoning:
The Original Question asks about a specific policy change from the Federal Transport Agency on a specific date (May 15, 2024). The Current Small Question targets this information. The provided document clearly states the policy change: the EV tax credit was reduced from $7,500 to $5,000. It also correctly attributes this to the "Federal Transport Agency". However, the document does not mention the date "May 15, 2024." This is a missing minor detail. The most important information—the actual policy change—is present and directly addresses the core of the Original Question. Therefore, I will extract this information.

### New Fact: The Federal Transport Agency announced an update to its electric vehicle subsidy program, reducing the maximum available tax credit for new purchases from $7,500 to $5,000.

Now think step by step based on the following information and generate the new fact
### Original Question: {question}
### Current Fact: {previous_cot}

### Current Small Question: {current_search_query}

### Relevant Documents for Small Question: {retrieved_documents_for_query}

### Reasoning:"""

    IR_COT_QUERY_GENERATOR_PROMPT = """You are an expert Query Generator. Your primary goal is to determine the *next logical piece of information* required to answer the 'Original Question', given the 'Current Fact' (a list of facts already established). Based on this, you will generate a targeted search query using the most appropriate search methods.

**Requirements:**
1.  Your output must consist of three sections: `### Reasoning for Next Query:`, `### Next Search Query:`, and `### Keyword Search:`.
2.  Follow the formatting of the examples provided for these sections.
3.  The `Next Search Query` is for semantic search and should be a concise, natural language query targeting the missing information.
4.  The `Keyword Search` should list specific keywords for lexical search (e.g., names, dates, sources), or "NO_NEED" if not necessary.
5.  If multiple keywords, separate them with commas.
6.  If the 'Current Fact' fully addresses the 'Original Question', set `Next Search Query` to "NO_QUERY_NEEDED" and `Keyword Search` to "NO_NEED".
7.  Always aim to break down complex questions into smaller, manageable queries to build the answer step-by-step.
8.  **CRITICAL FAILURE RECOVERY:** If the `Current Fact` contains `"No new fact"`, you must treat this as a signal that the previous query was too complex or unanswerable. Your immediate priority is to **decompose the problem further**. Generate a much simpler, more foundational query to unblock the process. For example, instead of asking for a comparison, ask for a single data point; instead of asking for a conclusion, ask for a definition; or focus on one entity instead of two; or focus on the content of question instead of datetime and source (avoid everything related to datetime and source).

**Instruction:**
1.  Analyze the 'Original Question' to understand the complete information required.
2.  Review the 'Current Fact' to identify what is already known and what remains missing. **Pay close attention to any "No new fact" entries.**
3.  If `"No new fact"` is present, follow the critical failure recovery instruction (Requirement #8) to formulate a simpler query.
4.  Otherwise, formulate a `Next Search Query` for semantic search to gather the next logical piece of information.
5.  Identify any specific terms (e.g., names, dates, sources) that require exact matching (`LEXICAL`), and list them in `Keyword Search`. If none are needed, set to "NO_NEED".
6.  Use `Keyword Search` only when necessary (e.g., to confirm a source, handle misspellings, or retrieve precise details like dates). Do not include keywords unless they enhance retrieval.
7.  If the 'Current Fact' is sufficient, set `Next Search Query` to "NO_QUERY_NEEDED" and `Keyword Search` to "NO_NEED".

**Examples:**

**Example 1:**
### Original Question: Between the claims made by 'FusionForward Inc.' in their investor briefing on February 20, 2024, regarding achieving 'net energy gain' in their latest fusion reactor prototype, and the independent assessment report published by the 'National Ignition Facility Review Panel' on May 5, 2024, which analyzed publicly available data and supplementary materials from FusionForward, was there consistency in the definition of 'net energy gain' used and the substantiation of the claim?
### Current Fact:
(empty)
### Reasoning for Next Query:
The Original Question is complex and requires multiple pieces of information. It involves comparing FusionForward's claims with the Review Panel's assessment. With no 'Current Fact', the first logical step is to retrieve FusionForward's specific claim from their February 20, 2024, briefing. This is a small, focused question about their definition of 'net energy gain' and any supporting data. A semantic search can capture the concept, while keywords ensure we target the correct company and date.
### Next Search Query:
What did FusionForward Inc. claim about achieving 'net energy gain' in their latest fusion reactor prototype in their investor briefing on February 20, 2024?
### Keyword Search:
"FusionForward Inc.", "February 20, 2024"

**Example 2:**
### Original Question: Considering the World Health Organization's (WHO) public health guidance on sugar consumption issued on March 4, 2015, and the subsequent statements made by the International Food & Beverage Alliance (IFBA) in their press release dated April 10, 2015, regarding industry commitments, was there a clear alignment or a notable divergence in the proposed timelines and targets for sugar reduction in products?
### Current Fact:
- On March 4, 2015, the WHO recommended that adults and children reduce their daily intake of free sugars to less than 10 percent of total energy intake, with a further reduction to below 5 percent for additional health benefits.
### Reasoning for Next Query:
The Original Question requires a comparison between WHO guidance and IFBA commitments. The 'Current Fact' has established the WHO's position. The missing piece is the IFBA's stance. The next logical step is to query for the specific details of the IFBA's press release from April 10, 2015, focusing on their proposed timelines and targets for sugar reduction. This will provide the necessary information for the comparison.
### Next Search Query:
What specific timelines or targets for sugar reduction did the International Food & Beverage Alliance (IFBA) propose in their press release from April 10, 2015?
### Keyword Search:
"International Food & Beverage Alliance", "IFBA", "April 10, 2015"

**Example 3:**
### Original Question:
Comparing ingredient lists, does 'Coca-Cola Classic' not list 'aspartame' while 'Coca-Cola Zero Sugar' does?
### Current Fact:
- According to the Coca-Cola Company’s website, Coca-Cola Zero Sugar includes aspartame and acesulfame potassium as sweeteners in its ingredient list.
- The ingredient list for Coca-Cola Classic, as per the Coca-Cola Company’s product labeling, includes carbonated water, high fructose corn syrup, caramel color, phosphoric acid, natural flavors, and caffeine, with no mention of aspartame.
### Reasoning for Next Query:
The Original Question requires a comparison of ingredient lists. The 'Current Fact' already confirms that Coca-Cola Zero Sugar contains aspartame and Coca-Cola Classic does not. All necessary information to answer the Original Question has been gathered. Therefore, no further queries are needed.
### Next Search Query:
NO_QUERY_NEEDED
### Keyword Search:
NO_NEED

**Example 4:**
### Original Question:
Comparing the current official FDA guidance on daily sodium intake for adults with 'HealthySnacks Inc.'s' nutritional labels for their 'Savory Bites,' do all listed serving sizes fall within the FDA's 'low sodium' claim threshold?
### Current Fact:
- The FDA's current guidance recommends adults limit sodium intake to less than 2,300 mg per day.
- No new fact
### Reasoning for Next Query:
The Original Question requires comparing the sodium in 'Savory Bites' to the FDA's 'low sodium' threshold. The presence of "No new fact" is a critical failure signal, indicating that the previous query (likely a complex one asking for all serving sizes and the threshold at once) was unsuccessful. I must now simplify my approach by breaking the problem down. Before I can compare the product's sodium content to the 'low sodium' threshold, I first need to establish what that threshold is. It's a foundational piece of information that is currently missing. Therefore, the next logical and much simpler query is to ask for the FDA's specific definition of a 'low sodium' claim. Also I need to avoid mention datetime and source.
### Next Search Query:
What is definition or criteria for a 'low sodium' claim on food packaging?
### Keyword Search:
"low sodium"

**Now, generate the next search query:**
### Original Question: {question}
### Current Fact: {current_cot}

### Reasoning for Next Query:"""

    FINAL_ANSWER_PROMPT = """You are an expert at synthesizing information from a given set of facts and providing concise final answers to a complex 'Original Question'.
You are given the 'Original Question' and the 'Current Fact' (which is a list of all facts established in previous steps).
Your goal is to directly answer the 'Original Question' based *only* on the provided 'Current Fact' list.

Requirements:
1.  Your output must consist of two sections: `### Reasoning for Final Answer:` and `### Final Answer:`.
2.  Follow the formatting of the examples provided for these sections.
3.  The `Final Answer` must be concise and directly address the 'Original Question'. It should be a synthesis of the provided facts, often resulting in a "Yes," "No," a specific name/entity.
4.  If the 'Current Fact' list does not contain sufficient information to directly and confidently answer all aspects of the 'Original Question', the `Final Answer` must be "Insufficient information." Do not make assumptions or infer beyond the provided facts.

Instruction:
1.  Carefully analyze the 'Original Question' to understand precisely what is being asked, including any comparisons, conditions, changes over time, or specific entities involved.
2.  Thoroughly review all entries in the 'Current Fact' list.
3.  Based on your analysis, write down your step-by-step `Reasoning for Final Answer`. This should explain how the 'Current Fact' entries collectively support the answer to the 'Original Question', or why they are insufficient. This might involve comparing facts from different sources, about different subjects, or from different time points mentioned in the Original Question.
4.  Formulate the `Final Answer`.
5.  If the `Current Fact` list does not specify the source and datetime, but contain all information to answer the `Original Question`, the `Confidence Score` should be punished lightly (as Source and datetime is not too important for the answer).

Example 1:
### Original Question: According to "Tech Chronicle" and "Gadget Today," which company, known for its "Vision" series AI chips, recently announced a partnership with "AutoDrive Inc." to develop autonomous vehicle systems?

### Current Fact:
- "Tech Chronicle" reports that "InnovateAI," maker of the "Vision" AI chip, has partnered with "AutoDrive Inc." for autonomous vehicle development.
- "Gadget Today" confirms that "InnovateAI" is collaborating with "AutoDrive Inc." on self-driving technology, leveraging InnovateAI's "Vision" chip architecture.
- "Future Motors" also announced a new EV model.

### Reasoning for Final Answer:
The Original Question asks to identify a company known for "Vision" AI chips that partnered with "AutoDrive Inc.," based on two specific sources.
Fact 1 ("Tech Chronicle") identifies "InnovateAI" as the maker of the "Vision" AI chip and states its partnership with "AutoDrive Inc."
Fact 2 ("Gadget Today") corroborates that "InnovateAI" is working with "AutoDrive Inc." and mentions the "Vision" chip.
Fact 3 is irrelevant to the question.
Both specified sources point to "InnovateAI."

### Final Answer: InnovateAI

Example 2:
### Original Question: Does the "BioHealth Journal" report from May 2023 indicate a different efficacy rate for "DrugX" in treating Condition Y compared to the findings published by "Pharma Weekly" in April 2023 for the same drug and condition?

### Current Fact:
- "Pharma Weekly" (April 2023) published phase 3 trial results for "DrugX" showing a 75 percent efficacy rate in treating Condition Y.
- "BioHealth Journal" (May 2023) reported on a meta-analysis of "DrugX" studies, concluding an overall efficacy rate of 72 percent for Condition Y, with a note on varying results in subgroups.
- "DrugX" is also being tested for Condition Z.

### Reasoning for Final Answer:
The Original Question asks if two reports from different sources and slightly different times indicate different efficacy rates for "DrugX" for Condition Y.
Fact 1 states "Pharma Weekly" reported a 75 percent efficacy rate.
Fact 2 states "BioHealth Journal" reported a 72 percent efficacy rate.
Since 75 percent is different from 72 percent, the reports indicate different (though potentially statistically similar, the question is about reported difference) efficacy rates. The question asks "Does... indicate a different... rate," implying a yes/no answer.

### Final Answer: Yes

Example 3:
### Original Question: Considering the "Global Economic Outlook" report from Q4 2022 which highlighted manufacturing growth in Country A, and the "Trade Winds Digest" Q1 2023 article discussing export tariffs in Country A, did Country A's economic policy landscape see any significant shifts affecting trade between these two reporting periods?

### Current Fact:
- The "Global Economic Outlook" (Q4 2022) noted a 5 percent growth in Country A's manufacturing sector due to strong domestic demand.
- "Trade Winds Digest" (Q1 2023) reported that Country A introduced new export tariffs on specific raw materials effective January 2023.
- Country B also saw manufacturing growth in Q4 2022.

### Reasoning for Final Answer:
The Original Question asks if there was a significant shift in Country A's economic policy affecting trade between Q4 2022 and Q1 2023, based on the two reports.
Fact 1 describes manufacturing growth in Q4 2022.
Fact 2 describes the introduction of new export tariffs in Q1 2023 (effective January 2023).
The introduction of new export tariffs represents a significant shift in economic policy affecting trade. The question asks "did... see any significant shifts," implying a yes/no answer.

### Final Answer: Yes

Example 4:
### Original Question: Based on the "Arctic Climate Report" detailing ice melt in 2022 and the "Wildlife Conservation Bulletin" discussing polar bear populations in 2023, what specific adaptive behavior in polar bears was directly attributed to the 2022 ice melt by both sources?

### Current Fact:
- The "Arctic Climate Report" (2022) documented record-low sea ice extent in the Beaufort Sea during the summer of 2022.
- The "Wildlife Conservation Bulletin" (2023) noted a decline in polar bear body conditions in the Beaufort Sea region and increased instances of long-distance swimming.
- The "Arctic Climate Report" also mentioned warming ocean temperatures.

### Reasoning for Final Answer:
The Original Question asks for a *specific adaptive behavior* in polar bears directly attributed to the 2022 ice melt by *both* sources.
Fact 1 ("Arctic Climate Report") discusses the 2022 ice melt but does not mention polar bear behavior.
Fact 2 ("Wildlife Conservation Bulletin") discusses polar bear conditions and behavior (long-distance swimming) in 2023, and mentions a decline in body conditions, but does not explicitly state that both sources attribute a *specific adaptive behavior* to the 2022 ice melt. The Bulletin links its observations to the Beaufort Sea region where ice melt was reported, but doesn't confirm the "Arctic Climate Report" also made this specific link to a behavior. We are missing the explicit attribution of a specific adaptive behavior by *both* sources to the 2022 ice melt.

### Final Answer: Insufficient information.

Now, based on the following information, generate the final answer:
### Original Question: {question}
### Current Fact: {current_cot}

### Reasoning for Final Answer:"""

    TERMINATION_FINAL_ANSWER_PROMPT = """You are an expert at synthesizing information from a given set of facts and providing concise final answers to a complex 'Original Question'.
You are given the 'Original Question' and the 'Current Fact' (which is a list of all facts established in previous steps).
Your goal is to directly answer the 'Original Question' based *only* on the provided 'Current Fact' list, and to provide a confidence score for your answer.

Requirements:
1.  Your output must consist of three sections: `### Reasoning for Final Answer:`, `### Final Answer:`, and `### Confidence Score:`.
2.  Follow the formatting of the examples provided for these sections.
3.  The `Final Answer` must be concise and directly address the 'Original Question'. It should be a synthesis of the provided facts. This can be a definitive "Yes," "No," a specific name/entity. 
4.  The `Confidence Score` must be an integer from 0 (no confidence/pure guess) to 5 (very high confidence/direct proof).
5.  If the 'Current Fact' list provides very little or no evidence, or highly contradictory evidence without a clear path to resolution, the `Final Answer` should be the most likely answer you can think of (short and concise) and the `Confidence Score` should be low. However, if there is suggestive evidence pointing towards a likely answer, even if not explicitly confirmed, provide that likely answer (which might be qualified, e.g., "Likely Yes") and use a moderate `Confidence Score` (e.g., 2, 3, or 4) to reflect the level of certainty. Do not make wild assumptions; inferences should be reasonably supported by the provided facts, with the confidence score indicating the strength of this inference.
6.  If the 'Current Fact' list provides really little evidence, always try to put a concise and short answer into `Final Answer` and the `Confidence Score` should be 0.
7.  Always avoid answer like 'No answer', 'No individual', 'Both yes and no'. You must be determined to answer the question with a proper answer (then low score if you are unsure).
8.  The `Final Answer` should be short and concise (no more than 3 words, contain Yes/No/Name).

Instruction:
1.  Carefully analyze the 'Original Question' to understand precisely what is being asked, including any comparisons, conditions, changes over time, or specific entities involved.
2.  Thoroughly review all entries in the 'Current Fact' list.
3.  Based on your analysis, write down your step-by-step `Reasoning for Final Answer`. This should explain how the 'Current Fact' entries collectively support the answer to the 'Original Question', or why they are insufficient. Explain how you arrived at your `Confidence Score`, linking it to the strength and directness of the evidence. This might involve comparing facts from different sources, about different subjects, or from different time points mentioned in the Original Question.
4.  Formulate the `Final Answer`.
5.  Assign a `Confidence Score` from 0 to 5.

Example 1:
### Original Question: According to "Tech Chronicle" and "Gadget Today," which company, known for its "Vision" series AI chips, recently announced a partnership with "AutoDrive Inc." to develop autonomous vehicle systems?

### Current Fact:
- "Tech Chronicle" reports that "InnovateAI," maker of the "Vision" AI chip, has partnered with "AutoDrive Inc." for autonomous vehicle development.
- "Gadget Today" confirms that "InnovateAI" is collaborating with "AutoDrive Inc." on self-driving technology, leveraging InnovateAI's "Vision" chip architecture.
- "Future Motors" also announced a new EV model.

### Reasoning for Final Answer:
The Original Question asks to identify a company known for "Vision" AI chips that partnered with "AutoDrive Inc.," based on two specific sources.
Fact 1 ("Tech Chronicle") identifies "InnovateAI" as the maker of the "Vision" AI chip and states its partnership with "AutoDrive Inc."
Fact 2 ("Gadget Today") corroborates that "InnovateAI" is working with "AutoDrive Inc." and mentions the "Vision" chip.
Fact 3 is irrelevant to the question.
Both specified sources directly point to "InnovateAI" and mention all key elements of the question. The information is consistent and directly answers the question.

### Final Answer: InnovateAI
### Confidence Score: 5

Example 2:
### Original Question: Does the "BioHealth Journal" report from May 2023 indicate a different efficacy rate for "DrugX" in treating Condition Y compared to the findings published by "Pharma Weekly" in April 2023 for the same drug and condition?

### Current Fact:
- "Pharma Weekly" (April 2023) published phase 3 trial results for "DrugX" showing a 75 percent efficacy rate in treating Condition Y.
- "BioHealth Journal" (May 2023) reported on a meta-analysis of "DrugX" studies, concluding an overall efficacy rate of 72 percent for Condition Y, with a note on varying results in subgroups.
- "DrugX" is also being tested for Condition Z.

### Reasoning for Final Answer:
The Original Question asks if two reports from different sources and slightly different times indicate different efficacy rates for "DrugX" for Condition Y.
Fact 1 states "Pharma Weekly" reported a 75 percent efficacy rate.
Fact 2 states "BioHealth Journal" reported a 72 percent efficacy rate.
Since 75 percent is different from 72 percent, the reports indicate different efficacy rates. The question asks "Does... indicate a different... rate," and the facts directly support a "Yes."

### Final Answer: Yes
### Confidence Score: 5

Example 3:
### Original Question: Considering the "Global Economic Outlook" report from Q4 2022 which highlighted manufacturing growth in Country A, and the "Trade Winds Digest" Q1 2023 article discussing export tariffs in Country A, did Country A's economic policy landscape see any significant shifts affecting trade between these two reporting periods?

### Current Fact:
- The "Global Economic Outlook" (Q4 2022) noted a 5 percent growth in Country A's manufacturing sector due to strong domestic demand.
- "Trade Winds Digest" (Q1 2023) reported that Country A introduced new export tariffs on specific raw materials effective January 2023.
- Country B also saw manufacturing growth in Q4 2022.

### Reasoning for Final Answer:
The Original Question asks if there was a significant shift in Country A's economic policy affecting trade between Q4 2022 and Q1 2023, based on the two reports.
Fact 1 describes manufacturing growth in Q4 2022 (context).
Fact 2 describes the introduction of new export tariffs in Q1 2023 (effective January 2023).
The introduction of new export tariffs represents a significant shift in economic policy affecting trade. This directly answers the question based on the provided facts.

### Final Answer: Yes
### Confidence Score: 5

Example 4:
### Original Question: Based on the "Arctic Climate Report" detailing ice melt in 2022 and the "Wildlife Conservation Bulletin" discussing polar bear populations in 2023, what specific adaptive behavior in polar bears was directly attributed to the 2022 ice melt by both sources?

### Current Fact:
- The "Arctic Climate Report" (2022) documented record-low sea ice extent in the Beaufort Sea during the summer of 2022.
- The "Wildlife Conservation Bulletin" (2023) noted a decline in polar bear body conditions in the Beaufort Sea region and increased instances of long-distance swimming.
- The "Arctic Climate Report" also mentioned warming ocean temperatures.

### Reasoning for Final Answer:
The Original Question asks for a *specific adaptive behavior* in polar bears directly attributed to the 2022 ice melt by *both* sources.
Fact 1 ("Arctic Climate Report") discusses the 2022 ice melt but does not mention polar bear behavior or attribute any behavior to the ice melt.
Fact 2 ("Wildlife Conservation Bulletin") discusses polar bear conditions and behavior (long-distance swimming) in 2023 and links it to the Beaufort Sea region, but it does not state that the "Arctic Climate Report" also made this specific attribution.
The core requirement that *both sources* attribute a *specific adaptive behavior* to the 2022 ice melt is not met. One source provides information on ice melt, the other on polar bear behavior, but the explicit link by both is missing.

### Final Answer: long-distance swimming
### Confidence Score: 1

Example 5:
### Original Question: Did the new "EcoGro" fertilizer, launched by "AgriCorp" in March 2023, contribute to the increased crop yields reported by Farmer Giles for his Q2 2023 harvest?

### Current Fact:
- "AgriCorp Press Release" (March 1, 2023): "AgriCorp launches 'EcoGro', a new organic fertilizer designed to boost crop yields significantly. Available for purchase from March 15th."
- "Farmer's Journal" (July 10, 2023): "Local farmer Giles reported a 20 percent increase in his Q2 2023 (April-June) corn harvest compared to his Q2 2022 harvest. He mentioned trying 'some new farming techniques' this year."
- "Weather Report Aggregator" (Q2 2023): "Q2 2023 saw exceptionally favorable weather conditions for agriculture in Farmer Giles' region, with optimal rainfall and sunshine."

### Reasoning for Final Answer:
The Original Question asks if "EcoGro" fertilizer contributed to Farmer Giles' increased Q2 2023 crop yields.
Fact 1 establishes "EcoGro" was launched in March 2023, designed to boost yields, and available for purchase before/during Q2.
Fact 2 confirms Farmer Giles had increased yields in Q2 2023 and mentioned trying "some new farming techniques." Using a new fertilizer like "EcoGro" aligns with this statement.
Fact 3 introduces a significant confounding variable: exceptionally favorable weather, which is known to boost crop yields.
Farmer Giles did not explicitly state he used "EcoGro." While its launch, purpose, and the farmer's mention of "new farming techniques" make it plausible that "EcoGro" was used and contributed to the yield increase, the favorable weather also provides a strong explanation for the increased yield.
The evidence is suggestive but not definitive regarding "EcoGro's" contribution. It's a reasonable inference that it might have contributed as part of the "new techniques," but this is not explicitly confirmed, and the weather's impact is also present.

### Final Answer: Yes
### Confidence Score: 3

Example 6:
### Original Question: Did the new "EcoGro" fertilizer, launched by "AgriCorp" in March 2023, contribute to the increased crop yields reported by Farmer Giles for his Q2 2023 harvest?

### Current Fact:
- The current documents do not provide helpful information about the new "EcoGro" fertilizer
- The current documents do not provide helpful information about the increased crop yields reported by Farmer Giles for his Q2 2023 harvest
- The current documents do not provide helpful information about the new farming techniques that Farmer Giles tried this year
- The current documents do not provide helpful information about the weather conditions in Farmer Giles' region in Q2 2023

### Reasoning for Final Answer:
The Original Question asks if "EcoGro" fertilizer contributed to Farmer Giles' increased Q2 2023 crop yields.
All fact leads to no answer.
As the intruction states, I should answer in proper manner, so I will put a 'No', the only important thing is to put a very low score for Confidence Score.

### Final Answer: No
### Confidence Score: 0

Now, based on the following information, generate the final answer:
### Original Question: {question}
### Current Fact: {current_cot}

### Reasoning for Final Answer:"""

    @classmethod
    def get_prompt(cls, prompt_name: str, **kwargs) -> str:
        """Get a formatted prompt template"""
        if hasattr(cls, prompt_name):
            template = getattr(cls, prompt_name)
            return template.format(**kwargs)
        else:
            raise ValueError(f"Prompt template '{prompt_name}' not found")
    
    @classmethod
    def list_available_prompts(cls) -> list:
        """List all available prompt templates"""
        return [attr for attr in dir(cls) if attr.endswith('_PROMPT') and not attr.startswith('_')] 