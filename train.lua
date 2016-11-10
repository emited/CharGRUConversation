
require 'optim'
local gru = require 'gru'
local model_utils = require 'model_utils'
local DataSetLoader = require 'DataSetLoader'
require 'Embedding.lua'


local opt = {
	filename = 'data/qa.csv',
	batchSize = 32,
	iterations = 10000,
	hiddenSize = 512,
	optimName = 'adam',
	optimState = {learningRate = 1e-4, learningRateDecay=1e-2},
	seed = 122,
	saveEvery = 100,
	saveFile = 'saves/gru_model4',
	printEvery = 1,
}
torch.manualSeed(opt.seed)

local loader = DataSetLoader.create(opt.filename, opt.batchSize)
local alphabetSize = loader.getAlphabetSize()
local x, y = loader:nextBatch()

local protos = {}
protos.embedX = Embedding(alphabetSize, opt.hiddenSize)
protos.embedY = Embedding(alphabetSize, opt.hiddenSize)
protos.encodeGRU = gru.GRU(opt.hiddenSize, opt.hiddenSize)
protos.decodeGRU = gru.GRU(opt.hiddenSize, opt.hiddenSize)
protos.softmax = nn.Sequential()
	:add(nn.Linear(opt.hiddenSize, alphabetSize))
	:add(nn.SoftMax())
protos.criterion = nn.CrossEntropyCriterion()

local params, gradParams = model_utils.combine_all_parameters(
	protos.encodeGRU,
	protos.decodeGRU,
	protos.softmax)
params:uniform(-.08, .08)

local clones = {}
for name, proto in pairs(protos) do
	print('cloning '..name..'...')
	clones[name] = model_utils.clone_many_times(proto, 101) --sequences are clamped at maximal length 41
end


function feval(new_params)
	
	if new_params ~= params then
		params:copy(new_params)
	end
	
	gradParams:zero()
	local x, y = loader:nextBatch()

	--forward pass

	--encoding
	local Tx = x:size(2)
	local embeddingsX = {}
	local hEncode = {[0] = torch.zeros(opt.batchSize, opt.hiddenSize)}
	for t=1, Tx do
		embeddingsX[t] = clones.embedX[t]:forward(x[{{}, t}]:clone())
		hEncode[t] = clones.encodeGRU[t]:forward{embeddingsX[t], hEncode[t-1]}
	end

	--decoding
	local Ty = y:size(2)
	local predictions = {}
	local embeddingsY = {[0] = torch.zeros(opt.batchSize, opt.hiddenSize)}
	local hDecode = {[0] = hEncode[Tx]:clone()}
	local loss = 0
	for t=1, Ty do
		embeddingsY[t] = clones.embedY[t]:forward(y[{{}, t}])
		hDecode[t] = clones.decodeGRU[t]:forward{embeddingsY[t-1], hDecode[t-1]}
		predictions[t] = clones.softmax[t]:forward(hDecode[t])
		loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
	end

	--backward pass

	--decoding
	local dhDecode = {}
	local dEmbeddingsY = {[Ty]=torch.zeros(opt.batchSize, opt.hiddenSize)}
	dhDecode[Ty] = torch.zeros(opt.batchSize, opt.hiddenSize)
	for t=Ty, 1, -1 do
		local dLdO = clones.criterion[t]:backward(predictions[t], y[{{}, t}])

		if t == Ty then
			dhDecode[t] = clones.softmax[t]:backward(hDecode[t], dLdO)
		else
			dhDecode[t]:add(clones.softmax[t]:backward(hDecode[t], dLdO))
		end
		dEmbeddingsY[t-1], dhDecode[t-1] = unpack(clones.decodeGRU[t]:backward({embeddingsY[t-1], hDecode[t-1]}, dhDecode[t]))
		clones.embedY[t]:backward(y[{{}, t}], dEmbeddingsY[t])
		
	end

	--encoding
	local dhEncoding = {[Tx] = dhDecode[1]}
	local dEmbeddingsX = {}
	for t=Tx, 1, -1 do
		dEmbeddingsX[t], dhEncoding[t-1] = unpack(clones.encodeGRU[t]:backward({embeddingsX[t], hEncode[t-1]}, dhEncoding[t]))
		clones.embedY[t]:backward(x[{{}, t}], dEmbeddingsX[t])
	end

	gradParams:clamp(-5, 5)
	return loss/Ty, gradParams
end


local lossPlot = {}
local gradNormPlot = {}
for i = 1, opt.iterations do

	local _, loss = optim[opt.optimName](feval, params, opt.optimState)

	--printing and evaluating
	if i % opt.printEvery == 0 then
        print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], gradParams:norm()))
        print(opt.optimState)
	end
	if i % opt.saveEvery == 0 then
		print('saving model...')
		torch.save(opt.saveFile..'_opt.t7')
		torch.save(opt.saveFile..'_epoch_'..i..'.t7', protos)
	end
end