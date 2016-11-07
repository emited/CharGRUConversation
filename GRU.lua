
require 'nn'
require 'nngraph'

local gru = {}

function gru.GRU(I, H)
	local input = nn.Identity()()
	local h1 = nn.Identity()()
	local z = nn.Sigmoid()(nn.CAddTable()({
		nn.Linear(I, H)(input),
		nn.Linear(H, H)(h1)}))
	local r = nn.Sigmoid()(nn.CAddTable()({
		nn.Linear(I, H)(input),
		nn.Linear(H, H)(h1)}))
	local h2 = nn.Tanh()(nn.CAddTable()({
		nn.Linear(I, H)(input),
		nn.CMulTable()({r, h1})}))
	local h = nn.CAddTable()({
		nn.CMulTable()({nn.AddConstant(1)(nn.MulConstant(-1)(z)), h1}),
		nn.CMulTable()({z, h2})})
	return nn.gModule({input, h1}, {h})
end

return gru