# IR COT
+ No search, no rerank
+ Retrieve and Reason
## Flow

1.  **Ingredients:**
    *   A **base retriever** that can take a query and return relevant paragraphs.
    *   An **LLM** capable of few-shot CoT generation
    *   A few **demonstration examples**: These are (Question, full CoT, supporting paragraphs) sets.

2.  **Initial Retrieval:**
    *   Take the input **Question (Q)**.
    *   Use Q as a query for the base retriever to get an initial set of K paragraphs (e.g., `docs_0`).

3.  **Iterative Interleaving Loop:**
    This loop alternates between a "Reason" step and a "Retrieve" step.

    *   **Step (i): "Reason" (Retrieval-Guided Reasoning / Extend CoT):**
        *   **Input to LLM:**
            *   The original Question (Q).
            *   All paragraphs collected so far (e.g., `docs_0` in the first iteration, then `docs_0 + docs_1`, etc.). These are "cumulated docs" as shown in Figure 1.
            *   The CoT sentences generated so far (empty in the first iteration).
            *   The few-shot demonstration examples.
        *   **Action:** The LLM generates the *next sentence* of the Chain-of-Thought. Let's call this `CoT_sentence_i`.
        *   **Example (Fig 1):**
            *   Q: "In what country was Lost Gravity manufactured?"
            *   After initial retrieval, LLM generates `CoT_sentence_1`: "The Lost Gravity was manufactured by Mack Rides."

    *   **Step (ii): "Retrieve" (CoT-Guided Retrieval / Expand Retrieved Information):**
        *   **Input to Base Retriever:** The *last generated CoT sentence* (`CoT_sentence_i`) is used as the new query.
        *   **Action:** The base retriever fetches K *new* paragraphs based on this CoT sentence. These are added to the collection of "cumulated docs."
        *   **Example (Fig 1):**
            *   Query for retrieval: "The Lost Gravity was manufactured by Mack Rides."
            *   Retrieved docs now include info about Mack Rides.
        *   The LLM then uses these *updated* "cumulated docs" for the next "Reason" step to generate `CoT_sentence_2`: "Mack Rides is a company from Germany."

4.  **Termination Conditions:**
    The iterative loop stops when:
    *   The generated CoT sentence contains an explicit answer phrase like "The answer is...".
    *   A maximum predefined number of reasoning steps is reached.

5.  **Output of IRCoT (Retrieval Result):**
    *   All the paragraphs collected throughout all the retrieval steps (the final "cumulated docs").

6.  **Final Question Answering (The "Reader"):**
    *   The collected paragraphs from IRCoT are now used as context.
    *   The original question is posed again to an LLM (either the same one or a different setup) along with this rich context.
    *   The LLM then generates the final answer, often again using CoT prompting (generating a full CoT from scratch with the improved context) or direct QA prompting (just outputting the answer).
