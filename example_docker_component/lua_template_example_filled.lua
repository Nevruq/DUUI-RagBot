-- DUUI Lua Communication Script
-- ============================================
-- Generated from Hugging Face Model JSON:
--   model_id: debajyotimaz/codemix_hate
--   model_type: bert
--   architectures: BertForSequenceClassification
--   inferred_task: text-classification
--   num_labels: 2
--   id2label: {"0": "Non-hateful", "1": "Hateful"}
-- ============================================

StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

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

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results["begins"] == nil or results["ends"] == nil then
        return
    end

    local cas = inputCas:getCas()
    local typeSystem = cas:getTypeSystem()

    -- TYPE_NAME: Derived from model_id "codemix_hate" -> HateSpeech
    local annotationType = typeSystem:getType("org.texttechnologylab.annotation.HateSpeech")
    if annotationType == nil then
        print("ERROR: Type 'org.texttechnologylab.annotation.HateSpeech' not found in TypeSystem!")
        return
    end

    -- FEATURE_DECLARATIONS: For text-classification with id2label
    -- Labels will be: "Non-hateful" or "Hateful"
    local labelFeature = annotationType:getFeatureByBaseName("label")
    local scoreFeature = annotationType:getFeatureByBaseName("score")

    -- Extract result arrays
    local begins = results["begins"]
    local ends = results["ends"]

    -- OUTPUT_EXTRACTIONS: Standard for text-classification
    local labels = results["labels"]
    local scores = results["scores"]

    -- Create annotations
    for i = 1, #begins do
        local annotation = cas:createAnnotation(annotationType, begins[i], ends[i])

        -- FEATURE_SETTERS: label=String, score=Double
        annotation:setStringValue(labelFeature, labels[i])
        annotation:setDoubleValue(scoreFeature, scores[i])

        cas:addFsToIndexes(annotation)
    end
end
