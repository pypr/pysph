<%def name="do_group(helper, g_idx, group)" buffered="True">
// ------------------------------------------------------------------
// Group${g_idx}
% for dest, (eqs_with_no_source, sources, all_eqs) in group.data.items():
// Destination ${dest}
% if all_eqs.has_initialize():
// Initialization for destination ${dest}
${helper.get_initialize_kernel(g_idx, group, dest, all_eqs)}
% endif

% if len(eqs_with_no_source.equations) > 0:
% if eqs_with_no_source.has_loop():
// Equations with no sources.
${helper.get_simple_loop_kernel(g_idx, group, dest, all_eqs)}
% endif
% endif

% for source, eq_group in sources.items():
// Source ${source}.
% if eq_group.has_loop():
${helper.get_loop_kernel(g_idx, group, dest, source, eq_group)}
% endif
% endfor

% if all_eqs.has_post_loop():
// post_loop for destination ${dest}
${helper.get_post_loop_kernel(g_idx, group, dest, all_eqs)}
% endif
// Finished destination ${dest}.
% if group.update_nnps:
<% helper.call_update_nnps(group) %>
% endif
% endfor
// Finished Group${g_idx}
// ------------------------------------------------------------------
</%def>

#define abs fabs
#define pi M_PI
${helper.get_header()}

% for g_idx, group in enumerate(helper.object.mega_groups):
% if len(group.data) > 0:
${do_group(helper, g_idx, group)}
% endif
% endfor  # (for g_idx, group ...)
