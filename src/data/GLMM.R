library(lme4)
library(MASS)

rcov = function(mod)
{
  v = VarCorr(mod)
  r = attr(v[[1]], "correlation")
  diag(r) = attr(v[[1]], "stddev")
  sdcor2cov(r)
}

df = VerbAgg
df$r2 <- ifelse(df$r2 == "Y", 1, 0)
df$id = 1:nrow(df)

y_true = df$r2
model = glmer(r2 ~ situ + btype + mode + (situ + btype + mode | id), data = df, family = binomial, control = glmerControl(check.nobs.vs.nlev = "ignore",
                                                                                                               check.nobs.vs.rankZ = "ignore",
                                                                                                               check.nobs.vs.nRE = "ignore",
                                                                                                               calc.derivs = FALSE,
                                                                                                               optCtrl = list(maxfun = 10000)))

prior = mvrnorm(10000, mu = c(0, 0, 0, 0, 0), Sigma = rcov(model))
pairs(prior, main = 'GLMM prior')

y_pred = predict(model, type = 'response')
conditional_likelihood = sum(dbinom(y_true, size = 1, prob = y_pred, log = TRUE))
p_resid = resid(model, type = 'pearson')
s_p_resid = sum(resid(model, type = 'pearson')**2)
d_resid = resid(model, type = 'deviance')
s_d_resid = sum(resid(model, type = 'deviance')**2)

X = model.matrix(model)
Y = df$r2
data = cbind(X,Y)
write.csv(data, 'VerbAgg.csv')