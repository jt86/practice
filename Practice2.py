n_top_feats=1
best_C_baseline = 1.0
best_C_SVM = 10.
best_gamma_SVM = 100.
best_gamma_baseline=0.01
best_C_SVM_plus = 2
best_C_star_SVM_plus = 20
best_gamma_SVM_plus = 4
best_gamma_star_SVM_plus = 40
k=5
chosen_params_file.write("\n\n" + str(n_top_feats) + " top features,fold " + str(k) + ",baseline," + str(
    best_C_baseline) + "," + str(best_gamma_baseline))
chosen_params_file.write("\n ,,SVM," + str(best_C_SVM) + "," + str(best_gamma_SVM))
chosen_params_file.write("\n  ,,SVM+," + str(best_C_SVM_plus) + "," + str(best_gamma_SVM_plus) + "," + str(
    best_C_star_SVM_plus) + "," + str(best_gamma_star_SVM_plus))


