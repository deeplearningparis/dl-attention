-- Options
local opt = lapp [[
Train an LSTM to make additions.

Options:
   --nEpochs        (default 30)     nb of epochs
   --batchSize      (default 1)      batch size
   --charDim        (default 30)     char vector dimensionality
   --hiddens        (default 30)     nb of hidden units
   --learningRate   (default 1)      learning rate
   --maxGradNorm    (default 1)      cap gradient norm
   --paramRange     (default .1)     initial parameter range
   --trainSplit     (default .9)     training set / validation set ratio
]]

-- Libs
local d = require 'autograd'
local util = require 'autograd.util'
local model = require 'autograd.model'
local _ = require 'moses'
local tablex = require('pl.tablex')

d.optimize(true)

-- Seed
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(123)

-- Load in PENN Treebank dataset
local sources, targets, dict = require('./load_data.lua')
local nTokens = #dict.id2char

-- LSTM encoder
local lstm1, params = model.RecurrentLSTMNetwork({
   inputFeatures = opt.charDim,
   hiddenFeatures = opt.hiddens,
   outputType = 'last',
})
-- LSTM decoder
local lstm2 = model.RecurrentLSTMNetwork({
   inputFeatures = opt.hiddens,
   hiddenFeatures = opt.hiddens,
   outputType = 'all',
}, params)

-- Use built-in nn modules (outside f)
local lsm = d.nn.LogSoftMax()
local lossf = d.nn.ClassNLLCriterion()

-- Complete trainable function:
local f = function(params, x, yi, y)
   -- Select word vectors
   x = util.lookup(params.words.W, x)
   yi = util.lookup(params.words.W, yi)
   -- Encode all inputs through LSTM layers:
   local h1 = lstm1(params[1], x)
   local h2 = lstm2(params[2], yi, h1)
   -- Flatten batch + temporal
   local nElements = torch.size(y, 1) * torch.size(y, 2)
   local h2f = torch.view(h2, nElements, opt.hiddens)
   local yf = torch.view(y, nElements)
   -- Linear classifier:
   local h3 = h2f * params[3].W + torch.expand(params[3].b, nElements, #dict.id2char)
   -- Log soft max
   local yhat = lsm(h3)
   -- Loss:
   local loss = lossf(yhat, yf)
   -- Return avergage loss
   return loss
end

-- Linear classifier params:
table.insert(params, {
   W = torch.Tensor(opt.hiddens, #dict.id2char),
   b = torch.Tensor(1, #dict.id2char),
})

-- Init weights + cast:
for i,weights in ipairs(params) do
   for k,weight in pairs(weights) do
      weights[k]:uniform(-opt.paramRange, opt.paramRange)
   end
end

-- Word dictionary to train:
local words = torch.Tensor(#dict.id2char, opt.charDim)
words:uniform(-opt.paramRange, opt.paramRange)
params.words = {W = words}

local nparameters = 0
for i, param in ipairs(_.flatten(params)) do nparameters = nparameters + param:nElement() end
print('vocabulary size: ' .. #dict.id2char)
print('number of parameters in the model: ' .. nparameters)

function trim(v)
   -- only trim batch size 1
   if v:size(1) > 1 then return v end
   local nonBlank = {}
   for i=1, v:size(2) do if v[1][i] ~= dict.char2id[' '] then table.insert(nonBlank, v[1][i]) end end
   return torch.Tensor(nonBlank):view(1, #nonBlank)
end

function getBatch(s, t, i)
   local x = trim(s:narrow(1, (i-1) * opt.batchSize + 1, opt.batchSize)):contiguous()
   local y = trim(t:narrow(1, (i-1) * opt.batchSize + 1, opt.batchSize)):contiguous()
   local yi = torch.Tensor(y:size()):fill(dict.char2id[' '])
   if y:size(2) > 1 then 
      yi:narrow(2, 2, y:size(2) - 1):copy(y:narrow(2, 1, y:size(2) - 1))
   end
   return x, yi, y
end

-- Split in and train/val data sets
local trainNum = math.floor(sources:size(1) * opt.trainSplit)
local trainSources = sources:narrow(1, 1, trainNum)
local valSources = sources:narrow(1, trainNum + 1, sources:size(1) - trainNum)
local trainTargets = targets:narrow(1, 1, trainNum)
local valTargets = targets:narrow(1, trainNum + 1, sources:size(1) - trainNum)

local trainBatchNum = math.floor(trainSources:size(1) / opt.batchSize)
local valBatchNum = math.floor(valSources:size(1) / opt.batchSize)

-- Train it
local lr = opt.learningRate
local valPerplexity = math.huge
local df =  d(f, { optimize = true })

for epoch = 1,opt.nEpochs do
   -- Train:
   print('\nTraining Epoch #'..epoch)
   local tloss = 0
   for i = 1, trainBatchNum do
      xlua.progress(i, trainBatchNum)
      -- Next sequences:
      local x, yi, y = getBatch(trainSources, trainTargets, i)
      -- Grads:
      local grads,loss = df(params, x, yi, y)
      -- Cap gradient norms:
      local norm = 0
      for i, grad in ipairs(_.flatten(grads)) do
         norm = norm + torch.sum(torch.pow(grad,2))
      end
      norm = math.sqrt(norm)
      if norm > opt.maxGradNorm then
         for i, grad in ipairs(_.flatten(grads)) do
            grad:mul( opt.maxGradNorm / norm )
         end
      end
      -- Update params:
      for k,vs in pairs(grads) do
         for kk,v in pairs(vs) do
            params[k][kk]:add(-lr, grads[k][kk])
         end
      end
      tloss = tloss + loss
   end

   local tperplexity = math.exp(tloss / trainBatchNum)
   print(string.format("Training perplexity = %.3f", tperplexity))
   tloss = 0

   local vloss = 0
   for i = 1, valBatchNum do
      local x, yi, y = getBatch(valSources, valTargets, i)
      local loss = f(params, x, yi, y)
      vloss = vloss + loss
   end

   local vperplexity = math.exp(vloss / valBatchNum)
   print(string.format("Validation perplexity = %.3f", vperplexity))
   vloss = 0

   -- Learning rate scheme:
   if vperplexity > valPerplexity or (valPerplexity - vperplexity)/valPerplexity < .001 then
      lr = lr / 2
      print(string.format("Decreasing learning rate to %.4f", lr))
   end
   valPerplexity = vperplexity
end
