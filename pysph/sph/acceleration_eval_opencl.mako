<%def name="do_group(helper, g_idx, group)" buffered="True">
% for dest, (eqs_with_no_source, sources, all_eqs) in group.data.items():
% if all_eqs.has_initialize():
// ------------------------------------------------------------------
// Initialization for destination ${dest}
${helper.get_initialize_kernel(g_idx, dest, all_eqs)}
% endif

% if len(eqs_with_no_source.equations) > 0:
% if eqs_with_no_source.has_loop():
// SPH Equations with no sources.
${helper.get_simple_loop_kernel(g_idx, dest, all_eqs)}
% endif
% endif

% for source, eq_group in sources.items():
// Source ${source}.
% if eq_group.has_loop():
${helper.get_loop_kernel(g_idx, dest, source, eq_group)}
% endif
% endfor

% if all_eqs.has_post_loop():
// post_loop for destination ${dest}
${helper.get_post_loop_kernel(g_idx, dest, all_eqs)}
% endif
// Finished destination ${dest}.
// ------------------------------------------------------------------
% endfor
</%def>

#define abs fabs
${helper.get_header()}

% for g_idx, group in enumerate(helper.object.mega_groups):
% if len(group.data) > 0:
${do_group(helper, g_idx, group)}
% endif
% endfor  # (for g_idx, group ...)