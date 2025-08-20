#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

#install.packages(c("ggplot2", "tidyverse", "wesanderson", "latex2exp"))
library(latex2exp)
library(tidyverse)
library(wesanderson)
library(RColorBrewer)
library(patchwork)

## Make sure this matches config.ANALYSIS_DIR
FIGURE_DIR = "./artifacts/example-output"


main <- function() {

    ###########################################
    ## make medical rate history comparisons ##
    ###########################################
    ratedf = read_csv(file.path(FIGURE_DIR, "medhist-ratedf-fixed-gaps.csv"))
    ratedf = make.fixed.age.gap.factors(ratedf)
    levels.to.plot = c("<-2", ">6")
    age.gap.factors = levels(ratedf$metric_q)
    n.factors = length(age.gap.factors)

    figure.dir = FIGURE_DIR
    output.dir = file.path(figure.dir, "medhist-rate-figs")
    dir.create(output.dir, showWarnings=FALSE)

    ## Print summaries of rel_risks
    ## roughly order conditions by effect size
    ratedf %>% filter(
        (biological_sex == "Female")
        &(variable %in% c("rel_risk"))#, "rate"))
        &(model %in% c("yhat-mod_global_weight_none-subset-all", "yhat-general-train-mod_global_weight_none-subset-all"))
        &(condition %in% c("diabetes"))
        &(age_bin %in% c("[25,35)", "[35,45)", "[45,55)", "[65,75)"))
        &(metric_q %in% c("all", levels.to.plot))
    ) %>% select(biological_sex, age_bin, metric_q, metric_group, variable, ci_str, condition, model) %>% print(n=100)
    #head(100)

    mod = "yhat-mod_global_weight_none-subset-all"
    cond.order = ratedf %>%
        filter(
            (metric_q == age.gap.factors[1])
            & (variable=="rel_risk") 
            & (model==mod)
            & (age_bin=="[45,55)")
        ) %>%
        group_by(condition) %>%
        summarize(rr=mean(q_mid)) %>%
        arrange(rr) %>%
        pull(condition)
    print(cond.order)

    for(sex in c("Male", "Female")) {
        for(cond in cond.order) {
            for(stat in c("rate", "rel_risk")) {
                make.medhist.plot(ratedf, sex, mod, cond, stat=stat)
                fname = file.path(output.dir, sprintf("rate-%s-%s-%s.pdf", cond, sex, stat=stat))
                ggsave(fname, width=6, height=2.5)
            }
        }
    }

    ## compare medhist high low all conditions, one chart
    output.dir = file.path(figure.dir, "medhist-comparison-figs")
    output.dir.hr = file.path(figure.dir, "medhist-comparison-figs-hr")
    output.dir.gen = file.path(figure.dir, "medhist-comparison-figs-healthy-general")
    dir.create(output.dir, showWarnings=FALSE)
    dir.create(output.dir.hr, showWarnings=FALSE)
    dir.create(output.dir.gen, showWarnings=FALSE)
    age_bins = ratedf$age_bin %>% unique
    co = cond.order[cond.order!="valvular_disease"]
    for(sex in c("Male", "Female")) {
        for(ab in age_bins) {
            make.condition.plot(ratedf, sex=sex, age.bin=ab, cond.order=co, mod=mod,
                                output.dir=output.dir, levels.to.plot=levels.to.plot)
            make.healthy.general.comparison.plot(
                ratedf,
                models = c("yhat-general-train-mod_global_weight_none-subset-all",
                           "yhat-mod_global_weight_none-subset-all"),
                sex=sex,
                age.bin=ab,
                cond.order=co,
                levels.to.plot=levels.to.plot,
                output.dir=output.dir.gen
            )
            make.healthy.general.comparison.plot(
                ratedf,
                models = c("yhat-hr-hrv-rf-baseline",
                           "yhat-mod_global_weight_none-subset-all"),
                sex=sex,
                age.bin=ab,
                cond.order=co,
                levels.to.plot=levels.to.plot,
                output.dir=output.dir.hr
            )
        }
    }

    ##################################
    ## VO2Max med hist comparisons  ##
    ##################################
    ratedf = read_csv(file.path(figure.dir, "medhist-vo2-ratedf.csv"))
    output.dir = file.path(figure.dir, "medhist-rate-vo2-figs")
    dir.create(output.dir, showWarnings=FALSE)
    mod = "yhat-mod_global_weight_none-subset-all"
    for(sex in c("Male", "Female")) {
        for(cond in cond.order) {
            for(age.bin in age_bins) {
                for(stat in c("rate", "rel_risk")) {
                    make.medhist.vo2.plot(ratedf, sex, mod, cond, age.bin, stat)
                    fname = file.path(output.dir, sprintf("rate-%s-%s-%s-%s.pdf", cond, sex, stat, age.bin))
                    ggsave(fname, width=6, height=2.5)
                }
            }
        }
    }

    ratedf %>% filter(
        (model == "yhat-mod_global_weight_none-subset-all")
        &(metric_q == "group")
        &(variable=="n_subj")
        &(condition=="diabetes")
        &(metric_group == "all")
        &(vo2max_q != -1)
    ) %>% select(q_mid) %>% sum


    ####################################
    ## make cleaned up smoking figure  #
    ####################################
    ydf = read_csv(file.path(figure.dir, "years-added-smoking-df-adj.csv"))
    metric.col = "gap_adj_spline" #"gap_adj"
    output.dir = file.path(figure.dir, "years-added-smoking-adj")
    dir.create(output.dir, showWarnings=FALSE)
    for(sex in c("Male", "Female")) {
        make.smoking.plots(ydf, sex=sex, metric.col = metric.col)
        fname = file.path(output.dir, sprintf("smoking-years-above-healthy-%s.pdf", sex))
        ggsave(fname, width=7, height=3.5)
    }


    #####################################
    # Make HR/HRV Comparison Figures    #
    #####################################
    ratedf = read_csv(file.path(figure.dir, "medhist-vo2-ratedf.csv"))
    errdf = read_csv(file.path(figure.dir, "fig-errors/error-global.csv"))
    output.dir = file.path(figure.dir, "medhist-comparison-figs-hr")
    make.hrv.error.comparison(errdf)
    ggsave(file.path(output.dir, "ppg-hr-hrv-error-comparison.pdf"), width=6, height=3)


    ###################################
    # Make error by demographic figs  #
    ###################################
    #for(cat in c("biological_sex", "bmiq", "age_bin", "race_eth")){
    f.dir = file.path(figure.dir, "fig-errors")
    x0 = make.demographic.error.figures(f.dir, "biological_sex", width=3.5)
    x1 = make.demographic.error.figures(f.dir, "bmiq", width=4.5)
    x2 = make.demographic.error.figures(f.dir, "race_eth", width=4.5)
    x3 = make.demographic.error.figures(f.dir, "age_bin", width=4.5)
    plt = x0 + x1 + x2 + x3 + 
        plot_layout(widths=c(3.5, 5, 6, 6.5))
    print(plt)
    out.dir = file.path(figure.dir, "error-by-demo")
    dir.create(out.dir, showWarnings=FALSE)
    ggsave(file.path(out.dir, "error-all.pdf"), width=14, height=2.8)

}


report.written.stat.comparisons <- function() {

    ratedf = read_csv(file.path(figure.dir, "medhist-ratedf-fixed-gaps.csv"))
    ratedf = make.fixed.age.gap.factors(ratedf)
    levels.to.plot = c(">6")
    ratedf = read_csv(file.path(figure.dir, "medhist-ratedf.csv"))
    levels.to.plot = c("q4") #c("q0", "q4")

    ## Print summaries of rel_risks
    ## roughly order conditions by effect size
    options(width=Sys.getenv("COLUMNS"))
    ratedf %>% filter(
        (biological_sex == "Male")
        &(variable %in% c("rel_risk"))#, "rate"))
        &(model %in% c("yhat-mod_global_weight_none-subset-all", "yhat-general-train-mod_global_weight_none-subset-all"))
        &(condition %in% c("diabetes")) #heartfailure")) #disease")) #, "heartfailure", "diabetes"))
        &(age_bin %in% c("[25,35)", "[35,45)", "[45,55)", "[55, 65]", "[65,75)"))
        &(metric_q %in% levels.to.plot)
    ) %>% select(biological_sex, age_bin, metric_q, metric_group, variable, ci_str, condition, model) %>% print(n=100)

}

make.sleep.var.figures <- function(figure.dir, variable, bio_sex, width=7) {
    ydf = read_csv(file.path(figure.dir, "sleep_vars.csv"))
    #variable = "first_rem_onset_hr_cat" #"deep_duration_hours_cat" #"sleep_efficiency_cat" #duration_hours_cat"
    pdf = ydf %>% filter(
        (hue == variable)
        &(sex == bio_sex)
    )
    pdf$q_mid <- as.numeric(pdf$mean)
    pdf$q_lo <- as.numeric(pdf$ci_lower)
    pdf$q_hi <- as.numeric(pdf$ci_upper)
    pdf$grp <- factor(as.numeric(pdf %>% pull(!!rlang::sym(variable))))
    #pdf$quin_codes <- as.numeric(pdf$quin_codes) + 1
    #pdf$quin_codes <- factor(pdf$quin_codes, 
    #                         levels=c(1, 2, 3, 4, 5))
    pdf$age_bin <- factor(
      pdf$age_cat,
      levels = c("[18, 25)", "[25, 35)", "[35, 45)", "[45, 55)", "[55, 65)",  "[65, 75)")
    )

    if(variable == "sleep_duration_hours_cat") {
        title = sprintf("Sleep duration (%s)", bio_sex)
        legend_title = "Sleep Dur. (quint)"
    }
    if(variable == "sleep_efficiency_cat") {
        title = sprintf("Sleep efficiency (%s)", bio_sex)
        legend_title = "Sleep Eff. (quint)"
    }
    if(variable == "deep_duration_hours_cat") {
        title = sprintf("Deep sleep duration (%s)", bio_sex)
        legend_title = "Deep Dur. (quint)"
    }
    if(variable == "first_rem_onset_hr_cat") {
        title = sprintf("REM latency(%s)", bio_sex)
        legend_title = "REM Latency (quint)"
    }


    ## plot
    dodge_width = .3
    y.lab <- "PpgAge Gap"
    x = ggplot() +
      geom_line(
        aes(x=age_bin,
            y=q_mid,
            group = grp,
            color=grp,
        ),
        data=pdf,
        position=position_dodge(dodge_width),
        show.legend=FALSE,
      ) +
      geom_point(
        aes(x=age_bin,
            y=q_mid,
            group=grp,
            color=grp,
        ),
        data=pdf,
        position=position_dodge(dodge_width)
      ) +
      geom_errorbar(
        aes(x=age_bin,
            y=q_mid,
            ymin=q_lo,
            ymax=q_hi,
            group=grp,
            color=grp,
        ),
        data=pdf,
        width=0,
        position=position_dodge(dodge_width)
      ) + 
      xlab("Chronological Age") + 
      ylab(y.lab) + 
      labs(color=legend_title) +
      ggtitle(title) + 
      scale_color_manual(
        values=wes_palette("Zissou1", 5, type="discrete")
      ) +
      theme_minimal()

    out.dir = file.path(figure.dir, "sleep-plots")
    dir.create(out.dir, showWarnings=FALSE)
    ggsave(file.path(out.dir, sprintf("age-gap-%s-%s.pdf", bio_sex, variable)), width=width, height=3.5)
}

make.demographic.error.figures <- function(figure.dir, variable, width) {
    edf = read_csv(file.path(figure.dir, sprintf("error-%s.csv", variable)))
    pdf = edf %>%
        filter(
            (model == "yhat-mod_global_weight_none-subset-all")
            &(split == "test")
        ) %>%
        mutate(
            cohort = if_else(is_healthy, "healthy", "general"),
        )
    print(pdf %>% select(is_healthy, !!rlang::sym(variable), age_mae_str))
    if(variable == "biological_sex") {
        leg.title = "Bio. Sex"
    }
    if(variable == "age_bin") {
        leg.title = "Chrono. age"
        pdf = pdf %>% mutate(
            age_bin = factor(age_bin, levels=c("<25", "[25,35)", "[35,45)", "[45,55)", "[55,65)", "[65,75)", ">75"))
        )
    }
    if(variable == "bmiq") {
        pdf = pdf %>% mutate(
            bmiq = factor(bmiq, levels=c("<18.5", "[18.5-25)", "[25-30)", ">=30"))
        )
        leg.title = "BMI"
    }
    if(variable == "race_eth"){
        pdf = pdf %>% mutate(
            race_eth = factor(race_eth, levels=c("Asian", "Black", "Hispanic", "White", "Multi", "other"))
        )
        leg.title = "Race/Ethnicity"
    }

    n.level = length(unique(pdf %>% pull(!!rlang::sym(variable))))

    ## plot
    dodge_width = .8
    y.lab <- "PpgAge Gap"
    x = ggplot() +
    geom_bar(
      aes(x=cohort,
          y=age_mae,
          group = !!rlang::sym(variable),
          fill = !!rlang::sym(variable),
      ),
      data=pdf,
      stat="identity",
      position=position_dodge(dodge_width),
      width=dodge_width*.95,
    ) +
    geom_errorbar(
      aes(x=cohort,
          y=age_mae,
          ymin=age_mae_lo,
          ymax=age_mae_hi,
          group = !!rlang::sym(variable),
          fill = !!rlang::sym(variable),
      ),
      data=pdf,
      width=0,
      position=position_dodge(dodge_width)
    ) +
    ylab("MAE (Years)") +
    scale_fill_manual(
      values=wes_palette("Zissou1", n.level, type="continuous")
    ) +
    theme_minimal() +
    guides(fill=guide_legend(title=leg.title))
    x
}


make.fixed.age.gap.factors <- function(ratedf) {
    ## fix metric_q
    ratedf = ratedf %>% mutate(metric_q = metric_group) %>% mutate(
        metric_q = str_replace(metric_q, fixed("(-inf, -2.0]"), "<-2"),
        metric_q = str_replace(metric_q, fixed("(-2.0, 0.0]"), "(-2, 0]"),
        metric_q = str_replace(metric_q, fixed("(0.0, 2.0]"), "(0, 2]"),
        metric_q = str_replace(metric_q, fixed("(2.0, 4.0]"), "(2, 4]"),
        metric_q = str_replace(metric_q, fixed("(4.0, 6.0]"), "(4, 6]"),
        metric_q = str_replace(metric_q, fixed("(6.0, inf]"), ">6"),
        metric_q = str_replace(metric_q, "group", "all"),
        metric_q = factor(metric_q, levels=c("all", "<-2", "(-2, 0]", "(0, 2]", "(2, 4]", "(4, 6]", ">6"))
    )
    ratedf
}



make.hrv.error.comparison <- function(errdf) {
    pdf = errdf %>% filter(
        (split == "test")
        &(biological_sex == "All")
    ) %>%
    mutate(
        model = str_replace(model, "yhat-mod_global_weight_none-subset-all", "full waveform"),
        model = str_replace(model, "yhat-general-train-mod_global_weight_none-subset-all", "general train cohort"),
        model = str_replace(model, "yhat-hr-hrv-baseline", "hr/hrv (linear)"),
        model = str_replace(model, "yhat-hr-hrv-rf-baseline", "hr/hrv (random forest)"),
        cohort = if_else(is_healthy, "healthy", "general"),
    )
    ## plot
    dodge_width = .5
    y.lab <- "Age Gap (above healthy)"
    x = ggplot() +
    geom_bar(
      aes(x=cohort,
          y=age_mae,
          group=model,
          fill=model,
      ),
      data=pdf,
      stat="identity",
      position=position_dodge(dodge_width),
      width=dodge_width*.95,
      #show.legend=FALSE,
    ) +
    geom_errorbar(
      aes(x=cohort,
          y=age_mae,
          ymin=age_mae_lo,
          ymax=age_mae_hi,
          group=model,
          fill=model,
      ),
      data=pdf,
      width=0,
      position=position_dodge(dodge_width)
    ) +
    ylab("Mean Abs. Error (Years)") +
    ggtitle("Full waveform vs HR/HRV prediction error") +
    scale_fill_manual(
      values=wes_palette("Zissou1", 5, type="continuous")
    ) +
    theme_minimal()
    #theme(legend.position = c(0.9, 0.7))
    print(x)
    x
}


make.smoking.plots <- function(
  ydf,
  sex,
  age_bins_to_filter = c(">75", "All"),
  #quintile_level_to_filter,
  title,
  metric.col="gap_adj"
  ){
  legend_title = "smoking status"
  title = sprintf("Years above healthy by smoking status (%s)", sex)
  ## prep data
  pdf = ydf %>%
    filter(
      (biological_sex == sex)
      & !(age_bin %in% age_bins_to_filter)
      & (smoking_status != "healthy")
    ) %>%
    select(-biological_sex) %>% 
    pivot_wider(
        id_cols = c("age_bin", "smoking_status"),
        names_from="var",
        values_from=metric.col
    )

  pdf$q_mid <- as.numeric(pdf$bmid)
  pdf$q_lo <- as.numeric(pdf$blo)
  pdf$q_hi <- as.numeric(pdf$bhi)
  #pdf$quin_codes <- as.numeric(pdf$quin_codes) + 1
  #pdf$quin_codes <- factor(pdf$quin_codes, 
  #                         levels=c(1, 2, 3, 4, 5))
  pdf$age_bin <- factor(
    pdf$age_bin, 
    levels = c("<25", "[25,35)", "[35,45)", "[45,55)", "[55,65)",  "[65,75)")
  )
  pdf$smoking_status <- factor(
    pdf$smoking_status,
    levels = c("never_smoker", "not_at_all", "some_days", "every_day")
  )

  ## plot
  dodge_width = .3
  y.lab <- "PpgAge Gap"
  x = ggplot() +
    geom_line(
      aes(x=age_bin, 
          y=q_mid, 
          group=smoking_status,
          color=smoking_status,
      ),
      data=pdf,
      position=position_dodge(dodge_width),
      show.legend=FALSE,
    ) +
    geom_point(
      aes(x=age_bin,
          y=q_mid,
          group=smoking_status,
          color=smoking_status,
      ),
      data=pdf,
      position=position_dodge(dodge_width)
    ) +
    geom_errorbar(
      aes(x=age_bin,
          y=q_mid,
          ymin=q_lo, 
          ymax=q_hi,
          group=smoking_status,
          color=smoking_status,
      ),
      data=pdf,
      width=0,
      position=position_dodge(dodge_width)
    ) + 
    xlab("Chronological Age") + 
    ylab(y.lab) + 
    labs(color=legend_title) +
    ggtitle(title) +
    scale_color_manual(
      values=wes_palette("Zissou1", 5, type="continuous")
    ) +
    theme_minimal() + 
    theme(legend.position = c(0.2, 0.7))

  print(x)
  return(x)
}



make.medhist.vo2.plot <- function(
    ratedf,
    sex,
    mod,
    cond,
    age.bin,
    stat="rate",
    age.bins.to.filter=c("<25", ">75")
) {

    title = sprintf("%s diagnosis rate (%s, age %s)", cond, tolower(sex), age.bin)
    y.lab = "Diagnosis Rate"
    if(stat == "rel_risk") {
        title = sprintf("%s relative risk (%s, age %s)", cond, tolower(sex), age.bin)
        y.lab = "Rel. Diagnosis Rate"
    }

    pdf = ratedf %>%
        filter(
            (biological_sex == sex)
            & (age_bin == age.bin)
            & (model==mod)
            & (variable == stat)
            & (condition == cond)
            & (vo2max_q != -1)
        ) %>%
        select(-biological_sex, -model, -variable) %>%
        mutate(age_bin = fct_relevel(age_bin, unique(age_bin)))
    ave_data = pdf %>% filter(metric_q == "group")
    q_data = pdf %>% filter(metric_q != "group")
    dodge_width = .4
    x = ggplot() +
        geom_line(
            aes(x=vo2max_q, y=q_mid, group=1),
            data=ave_data,
            color='grey',
            show.legend=FALSE,
        ) +
        geom_errorbar(
            aes(x=vo2max_q, ymin=q_lo, ymax=q_hi),
            width=0,
            data=ave_data,
            color = 'gray',
            show.legend=FALSE,
        ) +
        geom_point(
            aes(x = vo2max_q,
                y = q_mid,
                group=metric_q,
                color=metric_q,
                shape=metric_q,
            ),
            data=q_data,
            position=position_dodge(dodge_width)
        ) +
        geom_errorbar(
            aes(x = vo2max_q,
                y = q_mid,
                ymin =q_lo, ymax=q_hi,
                group=metric_q,
                color=metric_q,
                shape=metric_q,
            ),
            data=q_data,
            width=0,
            position=position_dodge(dodge_width)
        ) + 
        xlab(TeX("VO$_2$ max")) +
        ylab(y.lab) + 
        labs(color="Age Gap Q.", shape="Age Gap Q.") + 
        ggtitle(title) + 
        scale_color_manual(
            values=wes_palette("Zissou1", 5, type="continuous")
        ) +
        theme_minimal()

}

make.medhist.plot <- function(
    ratedf,
    sex,
    mod,
    cond,
    stat="rate",
    age.bins.to.filter=c("<25", ">75")
) {

    title = sprintf("%s diagnosis rate (%s)", cond, tolower(sex))
    y.lab = "Diagnosis Rate"
    if(stat == "rel_risk") {
        title = sprintf("%s relative risk (%s)", cond, tolower(sex))
        y.lab = "Rel. Diagnosis Rate"
    }

    leg.lab = "PpgAge Gap Q."
    if(">6" %in% levels(ratedf$metric_q)) {
        leg.lab = "PpgAge Gap (years)"
    }


    pdf = ratedf %>%
        filter(
            (biological_sex == sex)
            & !(age_bin %in% age.bins.to.filter)
            & (model==mod)
            & (variable == stat)
            & (condition == cond)
        ) %>%
        select(-biological_sex, -model, -variable) %>%
        mutate(age_bin = fct_relevel(age_bin, unique(age_bin)))
    ave_data = pdf %>% filter(metric_q == "all")
    q_data = pdf %>% filter(metric_q != "all")
    dodge_width = .4
    x = ggplot() +
        geom_line(
            aes(x=age_bin, y=q_mid, group=1),
            data=ave_data,
            color='grey',
            show.legend=FALSE,
        ) +
        geom_errorbar(
            aes(x=age_bin, ymin=q_lo, ymax=q_hi),
            width=0,
            data=ave_data,
            color = 'gray',
            show.legend=FALSE,
        ) +
        geom_point(
            aes(x = age_bin,
                y = q_mid,
                group=metric_q,
                color=metric_q,
                shape=metric_q,
            ),
            data=q_data,
            position=position_dodge(dodge_width)
        ) +
        geom_errorbar(
            aes(x = age_bin,
                y = q_mid,
                ymin =q_lo, ymax=q_hi,
                group=metric_q,
                color=metric_q,
                shape=metric_q,
            ),
            data=q_data,
            width=0,
            position=position_dodge(dodge_width)
        ) + 
        xlab("Age (Chrono.)") + 
        ylab(y.lab) + 
        labs(color=leg.lab, shape=leg.lab) + 
        ggtitle(title) + 
        scale_color_manual(
            values=wes_palette("Zissou1", 7, type="continuous")
        ) +
        theme_minimal()
}

#
# Make plot comparing effect sizes
#
make.condition.plot <- function(
    ratedf, sex, age.bin, cond.order, mod, output.dir, levels.to.plot=c("q0", "q4")
) {
    pdf = ratedf %>% filter(
        (biological_sex==sex)
        &(age_bin==age.bin)
        &(model==mod)
        &(variable=="rel_risk")
        &(condition %in% cond.order)
    )

    leg.lab = "Age Gap (years)"
    if("q0" %in% levels.to.plot){
        leg.lab = "Age Gap Q."
    }

    pdf = pdf %>%
        mutate(condition = fct_relevel(condition, cond.order)) %>%
        filter(metric_q %in% levels.to.plot)
    colors = wes_palette("Zissou1", 5, type="continuous")
    colors = c(colors[1], colors[5])
    ggplot() +
        geom_point(
            aes(
                x = condition,
                y = q_mid,
                color=metric_q,
            ),
            data=pdf
        ) +
        geom_errorbar(
            aes(
                x=condition,
                ymin=q_lo,
                ymax=q_hi,
                color=metric_q,
            ),
            data=pdf,
            width=0,
        ) + 
        scale_color_manual(values=colors) + 
        xlab("Condition") + 
        ylab("Rel. Risk") + 
        coord_flip() + 
        labs(color=leg.lab) +
        ggtitle(sprintf("Relative Diagnosis Rate, %s, %s", sex, age.bin)) +
        theme_minimal()
    ggsave(file.path(output.dir, sprintf("condition-summary-%s-%s.pdf", sex,age.bin)), width=6, height=4.5)
}


make.healthy.general.comparison.plot <- function(
    ratedf, models, sex, age.bin, cond.order, levels.to.plot, output.dir, height=4.5
) {
    pdf = ratedf %>% filter(
        (biological_sex==sex)
        &(age_bin==age.bin)
        &(variable=="rel_risk")
        &(condition %in% cond.order)
        &(model %in% models)
        &(metric_q %in% levels.to.plot)
    )
    pdf = pdf %>%
        mutate(
            condition = fct_relevel(condition, cond.order),
            model = str_replace(model, "yhat-mod_global_weight_none-subset-all", "PpgAge (healthy train)"),
            model = str_replace(model, "yhat-general-train-mod_global_weight_none-subset-all", "PpgAge (general train)"),
            model = str_replace(model, "yhat-hr-hrv-rf-baseline", "hr/hrv model (healthy train)"),
            model = factor(
                model,
                levels = c(
                    "PpgAge (healthy train)",
                    "PpgAge (general train)",
                    "hr/hrv model (healthy train)"
                ),
              )
        )
    colors = wes_palette("Zissou1", 5, type="continuous")
    colors = c(colors[1], colors[5])
    dodge = position_dodge(.6)
    ggplot() +
        geom_point(
            aes(
                x = condition,
                y = q_mid,
                color=metric_q,
                shape=model,
            ),
            data=pdf,
            position=dodge,
        ) +
        geom_errorbar(
            aes(
                x=condition,
                ymin=q_lo,
                ymax=q_hi,
                color=metric_q,
                shape=model,
            ),
            data=pdf,
            width=0,
            position=dodge
        ) + 
        scale_color_manual(values=colors) + 
        xlab("Condition") + 
        ylab("Rel. Risk") + 
        coord_flip() + 
        labs(color="Age Gap") +
        ggtitle(sprintf("Relative Diagnosis Rate, %s, %s", sex, age.bin)) +
        theme_minimal()
    print(pdf %>% filter(condition=="diabetes"))
    ggsave(file.path(output.dir, sprintf("healthy-general-comparison-%s-%s.pdf", sex,age.bin)), width=7, height=height)
}

main()
