StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream, parameters)
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = TopicUtils:getDocumentTextLength(inputCas)
    local selection_types = parameters["selection"]

    local selections = {}
    local selections_count = 1
    for selection_type in string.gmatch(selection_types, "([^,]+)") do
        local sentences = {}
        if selection_type == "text" then
            local s = { text = doc_text, begin = 0, ['end'] = doc_len }
            sentences[1] = s
        else
            local sentences_count = 1
            local clazz = Class:forName(selection_type);
            local sentences_it = JCasUtil:select(inputCas, clazz):iterator()
            while sentences_it:hasNext() do
                local sentence = sentences_it:next()
                local s = {
                    text = sentence:getCoveredText(),
                    begin = sentence:getBegin(),
                    ['end'] = sentence:getEnd()
                }
                sentences[sentences_count] = s
                sentences_count = sentences_count + 1
            end
        end

        local selection = { sentences = sentences, selection = selection_type }
        selections[selections_count] = selection
        selections_count = selections_count + 1
    end

    outputStream:write(json.encode({
        selections = selections,
        lang = doc_lang,
        doc_len = doc_len
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["begins"] ~= nil then
        local source = results["model_source"]
        local model_version = results["model_version"]
        local model_name = results["model_name"]
        local model_lang = results["model_lang"]

        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        local model_meta = luajava.newInstance("org.texttechnologylab.annotation.model.MetaData", inputCas)
        model_meta:setModelVersion(model_version)
        model_meta:setModelName(model_name)
        model_meta:setSource(source)
        model_meta:setLang(model_lang)
        model_meta:addToIndexes()

        local begins = results["begins"]
        local ends = results["ends"]

        -- MODEL_FIELDS are mapped below. Replace/extend as needed.
        -- Example: local labels = results["labels"]

        for index_i, _ in ipairs(begins) do
            local anno = luajava.newInstance("${ANNOTATION_CLASS}", inputCas)
            anno:setBegin(begins[index_i])
            anno:setEnd(ends[index_i])

            -- Field mapping here, e.g.:
            -- anno:setLabel(labels[index_i])

            -- Dynamic field mapping
            for key, value in pairs(results) do
                if type(value) == "table"
                    and #value > 0
                    and key ~= "begins"
                    and key ~= "ends"
                    and key ~= "modification_meta"
                    and key ~= "meta"
                    and key ~= "model_source"
                    and key ~= "model_version"
                    and key ~= "model_name"
                    and key ~= "model_lang"
                then
                    local v = value[index_i]
                    if v ~= nil then
                        local fieldName = key
                        local setterName = "set"
                            .. tostring(fieldName)
                                :gsub("^%l", function(c) return string.upper(c) end)
                                :gsub("_", "")

                        if type(anno[setterName]) == "function" then
                            anno[setterName](anno, v)
                        end
                    end
                end
            end

            anno:setModel(model_meta)
            anno:addToIndexes()
        end
    end
end
