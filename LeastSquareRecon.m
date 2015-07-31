%%% Description %%%
% This is a temporary measure to bridge python and matlab
% This matlab script is called through bash, and its arguments and results
% are passed through .mat files
% The name of the script, argument files and return files are hardcoded

%%% Parameters %%%
% Script Name: LeastSquareRecon.m
% Argument Files: slopes.mat, support_vectors.mat
% Return Files: surface.mat

%%% Dependencies %%%
% This script relies on g2s.m in the grad2Surf library, which in turn relies on
% the DOPbox library

%%% Usage %%%
% The path to the libraries are hardcoded. Change the locations of the libraries
% as need be
% Put this file in the same directory as the python scripts. This will make 
% IO of files more convenient

%%% Future %%%
% Find out how to use mlabwrap library

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('/home/tpl/MATLAB/grad2SurfV1_0/grad2SurfV1-0/grad2Surf');
addpath('/home/tpl/MATLAB/DOPBoxV1-8/DOPBoxV1-8/DOPbox');
import Grad2Surf.*

load('slopes.mat');
load('support_vectors.mat')

x = support_vectors(1,:,:);
y = support_vectors(2,:,:);

shape_x = size(x);
shape_y = size(y);

if shape_x(1) == 1
    x = double(transpose(x));
end
if shape_y(1) == 1
    y = double(transpose(y));
end

Zx = squeeze(slopes(1,:,:));
Zy = squeeze(slopes(2,:,:));

surface = g2s(Zx,Zy,x,y);

save('surface.mat','surface');
