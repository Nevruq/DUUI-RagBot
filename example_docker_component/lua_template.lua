-- DUUI Lua Communication Script Template
-- ============================================
-- LLM INSTRUCTIONS:
-- Generate a complete Lua script by replacing all {PLACEHOLDERS}
-- using the provided Hugging Face model JSON.
--
-- INPUT JSON FIELDS TO USE:
-- - model_id           -> For comments/documentation
-- - inferred_task      -> Determines script structure
-- - num_labels         -> Number of classification labels
-- - id2label           -> Label mapping {"0": "Label1", "1": "Label2", ...}
-- - model_type         -> For comments (e.g., "bert", "roberta")
-- - architectures      -> Determines output handling
--
-- OUTPUT STRUCTURE (for text-classification):
-- Python returns: {begins: [], ends: [], labels: [], scores: []}
-- Each label comes from id2label values
-- ============================================

StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

-- ============================================
-- SERIALIZE: Send text to Python for processing
-- THIS PART IS STATIC - no modification needed
-- ============================================
function serialize(inputCas, outputStream, parameters)
    local doc_text = inputCas:getDocumentText()
    local doc_len = string.len(doc_text)
    local doc_lang = inputCas:getDocumentLanguage()

    local selection_type = parameters["selection"]
    if selection_type == nil then
        selection_type = "text"
    end

    local sentences = {}

    if selection_type == "text" then
        local s = {
            text = doc_text,
            begin = 0,
            ['end'] = doc_len
        }
        sentences[1] = s
    else
        local cas = inputCas:getCas()
        local typeSystem = cas:getTypeSystem()
        local annotationType = typeSystem:getType(selection_type)

        if annotationType == nil then
            outputStream:write(json.encode({
                selections = {{sentences = {}, selection = selection_type}},
                lang = doc_lang,
                doc_len = doc_len
            }))
            return
        end

        local annotationIndex = cas:getAnnotationIndex(annotationType)
        local iterator = annotationIndex:iterator()
        local count = 1

        while iterator:hasNext() do
            local annotation = iterator:next()
            sentences[count] = {
                text = annotation:getCoveredText(),
                begin = annotation:getBegin(),
                ['end'] = annotation:getEnd()
            }
            count = count + 1
        end
    end

    outputStream:write(json.encode({
        selections = {{sentences = sentences, selection = selection_type}},
        lang = doc_lang,
        doc_len = doc_len
    }))
end

-- ============================================
-- DESERIALIZE: Process Python results into UIMA annotations
-- THIS PART IS DYNAMIC - modify based on model JSON
-- ============================================
function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results["begins"] == nil or results["ends"] == nil then
        return
    end

    local cas = inputCas:getCas()
    local typeSystem = cas:getTypeSystem()

    -- ========================================
    -- PLACEHOLDER: {TYPE_NAME}
    --
    -- GENERATION RULE:
    -- Based on inferred_task, use appropriate type:
    --   "text-classification" -> derive from model purpose
    --   Examples:
    --     - hate detection -> "org.texttechnologylab.annotation.HateSpeech"
    --     - sentiment     -> "org.texttechnologylab.annotation.Sentiment"
    --     - emotion       -> "org.texttechnologylab.annotation.Emotion"
    --
    -- INPUT USED: inferred_task, model_id (to infer purpose)
    -- ========================================
    local annotationType = typeSystem:getType("{TYPE_NAME}")
    if annotationType == nil then
        print("ERROR: Type '{TYPE_NAME}' not found in TypeSystem!")
        return
    end

    -- ========================================
    -- PLACEHOLDER: {FEATURE_DECLARATIONS}
    --
    -- GENERATION RULE:
    -- For text-classification (inferred_task == "text-classification"):
    --   ALWAYS generate these two features:
    --     local labelFeature = annotationType:getFeatureByBaseName("label")
    --     local scoreFeature = annotationType:getFeatureByBaseName("score")
    --
    -- The label values will be from id2label (e.g., "Non-hateful", "Hateful")
    -- The score is the confidence (0.0 to 1.0)
    --
    -- INPUT USED: inferred_task, id2label
    -- ========================================
    {FEATURE_DECLARATIONS}

    -- Extract result arrays from Python response
    local begins = results["begins"]
    local ends = results["ends"]

    -- ========================================
    -- PLACEHOLDER: {OUTPUT_EXTRACTIONS}
    --
    -- GENERATION RULE:
    -- For text-classification:
    --   local labels = results["labels"]
    --   local scores = results["scores"]
    --
    -- INPUT USED: inferred_task
    -- ========================================
    {OUTPUT_EXTRACTIONS}

    -- Create annotations for each result
    for i = 1, #begins do
        local annotation = cas:createAnnotation(annotationType, begins[i], ends[i])

        -- ========================================
        -- PLACEHOLDER: {FEATURE_SETTERS}
        --
        -- GENERATION RULE:
        -- For text-classification:
        --   annotation:setStringValue(labelFeature, labels[i])
        --   annotation:setDoubleValue(scoreFeature, scores[i])
        --
        -- Setter methods by type:
        --   String  -> setStringValue(feature, value)
        --   Double  -> setDoubleValue(feature, value)
        --   Integer -> setIntValue(feature, value)
        --   Boolean -> setBooleanValue(feature, value)
        --
        -- INPUT USED: inferred_task
        -- ========================================
        {FEATURE_SETTERS}

        cas:addFsToIndexes(annotation)
    end
end

-- ============================================
-- QUICK REFERENCE: JSON to Lua Mapping
-- ============================================
-- JSON Field              | Lua Usage
-- ------------------------|------------------------------------------
-- model_id                | Comment header only
-- inferred_task           | Determines feature structure
-- num_labels              | Validates label count (informational)
-- id2label                | Possible label values (e.g., "Hateful")
-- architectures[0]        | Determines output processing logic
-- recommended_postprocess | Confirms softmax + argmax approach
-- ============================================
