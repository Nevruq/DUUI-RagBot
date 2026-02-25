local Class = luajava.bindClass("java.lang.Class")
local JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
local StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
local DUUILuaUtils = luajava.bindClass("org.hucompute.duui.DUUILuaUtils")

-- Domain and utility types (best-effort bindings; environment dependent)
local Sentiment = luajava.bindClass("org.hucompute.textimager.uima.type.Sentiment")
local DocumentModification = luajava.bindClass("org.hucompute.textimager.uima.type.DocumentModification")
local MetaData = luajava.bindClass("org.hucompute.textimager.uima.type.MetaData")

-- Ensure json is available in the environment (json.encode / json.decode)
local json = json

-- Simple sentiment lexicons (tiny heuristic for demonstration)
local positiveWords = {
  ["good"]=true, ["great"]=true, ["happy"]=true, ["love"]=true, ["awesome"]=true,
  ["excellent"]=true, ["fantastic"]=true, ["wonderful"]=true, ["amazing"]=true, ["pleasant"]=true
}
local negativeWords = {
  ["bad"]=true, ["terrible"]=true, ["sad"]=true, ["hate"]=true, ["awful"]=true,
  ["worst"]=true, ["horrible"]=true, ["disappointing"]=true, ["boring"]=true, ["unpleasant"]=true
}

local function toWords(text)
  local words = {}
  if not text then return words end
  for w in string.gmatch(text:lower(), "%w+") do
    table.insert(words, w)
  end
  return words
end

local function computeScores(text)
  local posCount, neuCount, negCount = 0, 0, 0
  for _, w in ipairs(toWords(text)) do
    if positiveWords[w] then
      posCount = posCount + 1
    elseif negativeWords[w] then
      negCount = negCount + 1
    else
      neuCount = neuCount + 1
    end
  end
  local total = posCount + neuCount + negCount
  if total > 0 then
    return posCount/total, neuCount/total, negCount/total
  else
    -- Fallback: neutral if no content
    return 0.0, 1.0, 0.0
  end
end

function serialize(inputCas, outputStream, params)
  -- Extract core document data
  local docText = inputCas:getDocumentText()
  local docLang = inputCas:getDocumentLanguage()
  local docLen = 0
  if docText ~= nil then
    docLen = string.len(docText)
  end

  -- Handle optional text segmentation / selection_types for analysis
  local analysisText = docText
  if params and type(params) == "table" and params.selection_types then
    local parts = {}
    -- Try to collect text spans for each requested type
    for _, typeName in ipairs(params.selection_types) do
      local ok, klassOrErr = pcall(function()
        return luajava.bindClass(typeName)
      end)
      if ok and klassOrErr then
        local klass = klassOrErr
        local iterUtil = JCasUtil:select(inputCas, klass):iterator()
        while iterUtil:hasNext() do
          local a = iterUtil:next()
          if a and a:getBegin() ~= nil and a:getEnd() ~= nil then
            local b = a:getBegin()
            local e = a:getEnd()
            -- Conform to 0-based begin index; Lua string is 1-based
            local segment = ""
            if docText ~= nil and b >= 0 and e > b then
              segment = docText:sub(b + 1, e)
            end
            if segment and segment ~= "" then
              table.insert(parts, segment)
            end
          end
        end
      end
    end
    if #parts > 0 then
      analysisText = table.concat(parts, " ")
    end
  end

  -- Compute simple sentiment scores on analysisText
  local p, neu, ng = computeScores(analysisText)

  -- Build output JSON with document metadata and scores
  local output = {
    documentText = docText,
    documentLanguage = docLang,
    documentLength = docLen,
    sentimentScores = {
      positive = p,
      neutral = neu,
      negative = ng
    }
  }

  if json and type(json.encode) == "function" then
    local encoded = json.encode(output)
    if encoded then
      outputStream:write(encoded)
      return
    end
  end

  -- Fallback in case json encoding is not available
  outputStream:write("{\"positive\": " .. tostring(p) ..
                   ",\"neutral\": " .. tostring(neu) ..
                   ",\"negative\": " .. tostring(ng) .. "}")
end

function deserialize(inputCas, inputStream)
  -- Read JSON from inputStream
  local jsonText
  local okRead, rawBytes = pcall(function()
    return inputStream:readAllBytes()
  end)

  if okRead and rawBytes then
    -- Convert bytes to Java String then to Lua string
    local StringClass = luajava.bindClass("java.lang.String")
    local jsonStringObj = luajava.newInstance("java.lang.String", rawBytes, StandardCharsets.UTF_8)
    jsonText = tostring(jsonStringObj:toString())
  else
    -- If read failed, try to read as a direct string
    jsonText = ""
  end

  local data = nil
  if jsonText ~= nil and jsonText ~= "" and json and type(json.decode) == "function" then
    local status, result = pcall(function() return json.decode(jsonText) end)
    if status then
      data = result
    end
  end

  -- Fallback: ensure data is a table
  if not data or type(data) ~= "table" then
    data = {}
  end

  local beginPos = 0
  local endPos = 0
  local docLength = 0
  local docText = inputCas:getDocumentText() or ""

  if inputCas:getDocumentText() then
    endPos = inputCas:getDocumentText():len()
  end

  if data.begin ~= nil then
    beginPos = tonumber(data.begin)
  end
  if data.end ~= nil then
    endPos = tonumber(data.end)
  end
  if data.documentLength ~= nil then
    docLength = tonumber(data.documentLength)
  else
    docLength = docText:len()
  end

  -- Create and populate domain-specific sentiment annotations
  local sentAnnotations = {}

  local scores = {}
  if type(data.sentimentScores) == "table" then
    local sp = tonumber(data.sentimentScores.positive) or 0
    local sn = tonumber(data.sentimentScores.neutral) or 0
    local sg = tonumber(data.sentimentScores.negative) or 0
    scores = { positive = sp, neutral = sn, negative = sg }
  else
    -- If no scores provided, attempt to infer from data fields
    local sp = tonumber(data.positive) or 0
    local sn = tonumber(data.neutral) or 0
    local sg = tonumber(data.negative) or 0
    scores = { positive = sp, neutral = sn, negative = sg }
  end

  -- Helper to safely create and index a Sentiment annotation
  local function makeSentiment(beginA, endA, value)
    local ok, ann = pcall(function()
      return luajava.newInstance("org.hucompute.textimager.uima.type.Sentiment", inputCas)
    end)
    if ok and ann then
      local b = beginA or 0
      local e = endA or (docLength > 0 and docLength or 0)
      ann:setBegin(b)
      ann:setEnd(e)
      ann:setSentiment(tonumber(value) or 0)
      ann:addToIndexes()
      table.insert(sentAnnotations, ann)
    end
  end

  -- Create 3 sentiment annotations spanning the full document (per-value)
  local sharedBegin = 0
  local sharedEnd = docLength
  if docLength == 0 then
    sharedEnd = inputCas:getDocumentText():len()
  end
  makeSentiment(sharedBegin, sharedEnd, scores.positive or 0)
  makeSentiment(sharedBegin, sharedEnd, scores.neutral or 0)
  makeSentiment(sharedBegin, sharedEnd, scores.negative or 0)

  -- Create DocumentModification and MetaData annotations as metadata (best-effort)
  local okMod, docModAnn = pcall(function()
    return luajava.newInstance("org.hucompute.textimager.uima.type.DocumentModification", inputCas)
  end)
  if okMod and docModAnn then
    -- Attempt to set a minimal range covering the document
    local beginM = 0
    local endM = (inputCas:getDocumentText() and #inputCas:getDocumentText()) or 0
    if docLength > 0 then
      endM = docLength
    elseif inputCas:getDocumentText() then
      endM = inputCas:getDocumentText():len()
    end
    if docModAnn.setBegin then docModAnn:setBegin(beginM) end
    if docModAnn.setEnd then docModAnn:setEnd(endM) end
    if docModAnn.addToIndexes then docModAnn:addToIndexes() end
  end

  local okMeta, metaAnn = pcall(function()
    return luajava.newInstance("org.hucompute.textimager.uima.type.MetaData", inputCas)
  end)
  if okMeta and metaAnn then
    if metaAnn:setDocumentLanguage then
      metaAnn:setDocumentLanguage(inputCas:getDocumentLanguage())
    end
    if metaAnn:setDocumentText then
      metaAnn:setDocumentText(inputCas:getDocumentText())
    end
    if metaAnn:addToIndexes then metaAnn:addToIndexes() end
  end
end