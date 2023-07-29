#' Convert Post-Hoc Test Results to CLD
#'
#' Convert p values from pairwise comparisons to CLD.
#'
#' CLD - compact letter display.
#' This function is a wrapper around [multcompView::multcompLetters()].
#'
#' @note
#' No hyphens are allowed in group names
#' (vaues of culumns `group1` and `group2`).
#'
#' @param .data (data frame with at least 3 columns)
#'        The result of pairwise comparison test usually from \pkg{rstatix}
#'        package.
#' @param group1,group2 Name of the columns in `.data`, which contain the names
#'        of first and second group. Defaults to "group1" and "group2".
#' @param p_name Name of the column, which contains p values.
#'        Defaults to `p.adj`.
#' @param alpha Significance level. Defaults to 0.05.
#'
#' @return Data frame with compared group names and CLD representation of
#'         test results. Contains columns with group names and CLD results
#'         (`cld` and `spaced_cld`).

convert_pairwise_p_to_cld <- function(.data,
                                      group1 = "group1",
                                      group2 = "group2",
                                      p_name = "p.adj",
                                      output_gr_var = "group",
                                      alpha = 0.05) {

  # Checking input
  col_names <- c(group1, group2, p_name)
  missing_col <- !col_names %in% colnames(.data)

  if (any(missing_col)) {
    stop(
      "Check you input as these columns are not present in data: ",
      paste(col_names[missing_col], sep = ",")
    )
  }

  # Analysis
  pair_names <- stringr::str_glue("{.data[[group1]]}-{.data[[group2]]}")

  # Prepare input data
   purrr::set_names(.data[[p_name]], pair_names) |>
    # Get CLD
    multcompView::multcompLetters(threshold = alpha) |>
    # Format the results
    with(
      dplyr::full_join(
        Letters |>
          tibble::enframe(output_gr_var, "cld"),
        monospacedLetters |>
          tibble::enframe(output_gr_var, "spaced_cld") |>
          dplyr::mutate(
            spaced_cld = stringr::str_replace_all(spaced_cld, " ", "_")
          ),
        by = output_gr_var
      )
    )
}
