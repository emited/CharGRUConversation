
require 'optim'
require 'nn'
require 'Embedding'
require 'nngraph'
local model_utils = require 'model_utils'
local num2char = require 'num2char'
local char2num = require 'char2num'

local opt = {
	seed = 123,
	model = 'saves/gru_model3_epoch_2300.t7',
	trainOpt = '',
	text = 'hello how are you?',
	sample = false,
}

torch.manualSeed(opt.seed)


local protos = torch.load(opt.model)
--local trainOpt = torch.load(opt.trainOpt)
local trainOpt = {hiddenSize=512}

--forward of encoder
local hEncode = {[0] = torch.zeros(1, trainOpt.hiddenSize)}
local t = 1
for c in opt.text:gmatch'.' do
	local x = torch.Tensor({char2num[c]})
	local embeddingX = protos.embedX:forward(x)
	hEncode[t] = protos.encodeGRU:forward{embeddingX, hEncode[t-1]}
	t = t + 1
end

--forward of decoder
local hDecode = {[0]=hEncode[#hEncode]}
local predictions = {}
local probabilities = {}
local charPredictions = {}
local embeddingY = {[0]=torch.zeros(trainOpt.embeddingSize)}
for t = 1, 100 do
	
	--xlua.progress(t, 100)
	hDecode[t] = protos.decodeGRU:forward{embeddingY[t-1], hDecode[t-1]}
	probabilities[t] = protos.softmax:forward(hDecode[t])
	embeddingY[t] = probabilities[t]
	print(embeddingY)

	if opt.sample then
		predictions[t] = torch.multinomial(probabilities[t]:view(1, -1))
	else --argmax
		_, predictions[t] = probabilities[t]:view(1, -1):max(2)
	end
	--print(probabilities[1]:view(1, -1):max(2))
	print(predictions[t][1][1])
	print('ok')
	print(num2char[27])
	charPredictions[t] = num2char[predictions[t][1][1]]
	print(charPredictions)
end

print('reply = ')
print(table.concat(charPredictions))