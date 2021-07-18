function [resu,fold] = proc_argv(params)
    resu=params;
    fold=5;
    if ~isempty(params) && ~isempty(regexp(params,'-v'))
        [fold_val,fold_start,fold_end] = regexp(params,'-v\s+\d+','match','start','end');
        if ~isempty(fold_val)
            [temp1,fold] = strread([fold_val{:}],'%s %u');
            resu([fold_start:fold_end]) = [];
        else
            error('Number of CV folds must be specified by "-v cv_fold"');
        end
    end
end