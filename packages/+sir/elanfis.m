function fis = elanfis(train_x, train_y, n_mfs, epochs)
    % Extreme Learning ANFIS
    %
    % Parameters
    % ----------
    % `train_x`
    % 	Training data (X)
    % `train_y`
    %   Training data target (Y)
    % `n_mfs`
    %   Number of membership functions to use in all input.
    % `epochs`
    %   Number of random trials
    %
    % Returns
    % -------
    % `fis`
    %   The tuned TSK fuzzy inference system

    % Setting defaults

    if nargin <= 4
        epochs = 50;
    end
    if nargin <= 3
        n_mfs = 2;
    end

    [n_observations, n_variables] = size(train_x);
    n_outputs = size(train_y, 2);
    n_rules = n_mfs ^ n_variables;

    range_observations = range(train_x);
    min_observations = min(train_x);

    diff_c = range_observations / (n_mfs - 1);

    % Memory allocations
    w_inp = zeros(n_variables + 1, n_rules);
    total_w_inp = zeros(n_observations, (n_variables + 1) * n_rules);
    predictions = zeros(size(train_y));
    mf_fire = zeros(n_variables, n_mfs);
    rule_fire = zeros(n_variables, n_rules);

    optimum_error = -1;

    for ep = 1 : epochs

        % Settings parameters for bell membership functions
        for inp_i = 1 : n_variables
            for mf_i = 1 : n_mfs
                % For range 0 to 2 aj*
                % a(inp_i, mf_i) = 2 * rand * range_observations(inp_i) / (2 * (n_mfs - 1));
                % For range 0.5 to 1.5 aj*
                a(inp_i, mf_i) = (1.5 - rand) * range_observations(inp_i) / (2 * (n_mfs - 1));
                b(inp_i, mf_i) = 1.9 + (rand * 0.2);
                c(inp_i, mf_i) = ((rand - 0.5) * diff_c(inp_i)) + (min_observations(inp_i) + (mf_i - 1) * diff_c(inp_i));
            end
        end

        for obs_i = 1 : n_observations
            % Membership function firing strengths
            for inp_i = 1 : n_variables
                for mf_i = 1 : n_mfs
                    mf_fire(inp_i, mf_i) = 1 / (1 + (abs((train_x(obs_i, inp_i) - c(inp_i, mf_i)) / a(inp_i, mf_i))) ^ (2 * b(inp_i, mf_i)));
                end
            end

            % Rule firing strengths
            % 00000000000000000000000
            for inp_i = 1 : n_variables
                count = 1;
                for k = 1 : n_mfs ^ (inp_i - 1)
                    for mf_i = 1 : n_mfs
                        for l = 1 : n_mfs ^ (n_variables - inp_i)
                            rule_fire(inp_i, count) = mf_fire(inp_i, mf_i);
                            count = count + 1;
                        end
                    end
                end
            end
            % 00000000000000000000000

            weights = prod(rule_fire);
            weights_n = weights / sum(weights); % Normalized

            % Weighted inputs
            for inp_i = 1 : n_variables
                w_inp(inp_i, :) = train_x(obs_i, inp_i) * weights_n;
            end

            w_inp(n_variables + 1, :) = weights_n;

            total_w_inp(obs_i, :) = reshape(w_inp, 1, []);
        end

        % Learning parameters

        for out_i = 1 : n_outputs
            out(out_i).params = total_w_inp \ train_y(:, out_i);
        end

        % Error checking
        for out_i = 1 : n_outputs
            predictions(:, out_i) = total_w_inp * out(out_i).params;
        end

        error = sqrt(sum(sum((predictions - train_y) .^ 2)));

        if ep == 1
            optimum_error = error;
            optimum_a = a;
            optimum_b = b;
            optimum_c = c;
            optimum_out = out;
        elseif error < optimum_error
            optimum_error = error;
            optimum_a = a;
            optimum_b = b;
            optimum_c = c;
            optimum_out = out;
        end
    end

    % Formatting output parameters
    for out_i = 1 : n_outputs
        optimum_out(out_i).params_reshaped = reshape(optimum_out(out_i).params, n_variables + 1, n_rules)';
    end

    % Generating fis using optimum parameters

    fis = genfis1([train_x, train_y(:, 1)], n_mfs, 'gbellmf');
    fis.name = 'elanfis';

    % Setting optimum input parameters
    for inp_i = 1 : n_variables
        for mf_i = 1 : n_mfs
            fis.input(inp_i).mf(mf_i).params = [optimum_a(inp_i, mf_i), optimum_b(inp_i, mf_i), optimum_c(inp_i, mf_i)];
        end
    end

    % Preparing output mfs
    for out_i = 1 : n_outputs
        fis.output(out_i).name = ['output', num2str(out_i)];
        fis.output(out_i).range = [min(train_y(:, out_i)), max(train_y(:, out_i))];
        out_str = num2str(out_i);
        for rule_i = 1 : n_rules
            mf_str = num2str(rule_i);
            fis.output(out_i).mf(rule_i).name = ['out', out_str, 'mf', mf_str];
            fis.output(out_i).mf(rule_i).type = 'linear';
        end
    end

    % Rules
    for rule_i = 1 : n_rules
        for out_i = 1 : n_outputs
            fis.output(out_i).mf(rule_i).params = optimum_out(out_i).params_reshaped(rule_i, :);
        end
        fis.rule(rule_i).consequent = repmat(fis.rule(rule_i).consequent, 1, n_outputs);
    end

end
