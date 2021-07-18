function auc = plotroc(y,x,params)
%plotroc draws the recevier operating characteristic(ROC) curve.
%
%auc = plotroc(training_label, training_instance [, libsvm_options -v cv_fold])
%  Use cross-validation on training data to get decision values and plot ROC curve.
%
%auc = plotroc(testing_label, testing_instance, model)
%  Use the given model to predict testing data and obtain decision values
%  for ROC
%
% Example:
%  
%   load('heart_scale.mat');
%   plotroc(heart_scale_label, heart_scale_inst,'-v 5');
%
%   [y,x] = libsvmread('heart_scale');
%       model = svmtrain(y,x);
%   plotroc(y,x,model);
    rand('state',0); % reset random seed
    if nargin < 2
        help plotroc
        return
    elseif isempty(y) | isempty(x)
        error('Input data is empty');
    elseif sum(y == 1) + sum(y == -1) ~= length(y)
        error('ROC is only applicable to binary classes with labels 1, -1'); % check the trainig_file is binary
    elseif exist('params') && ~ischar(params)
        model = params;
        [predict_label,mse,deci] = svmpredict(y,x,model) ;% the procedure for predicting
        auc = roc_curve(deci*model.Label(1),y);
    else
        if ~exist('params')
            params = [];
        end
        [param,fold] = proc_argv(params); % specify each parameter
        if fold <= 1
            error('The number of folds must be greater than 1');
        else   
            [deci,label_y] = get_cv_deci(y,x,param,fold); % get the value of decision and label after cross-calidation
            auc = roc_curve(deci,label_y); % plot ROC curve
        end
    end
end