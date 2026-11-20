"""Human-readable console view; the saved JSON keeps raw SI units."""


def format_report(report, path):
    model = path.stem.removesuffix('_pos_y').removesuffix('_neg_y')
    total = report['episodes']
    lines = [
        f"=== {model} | {report['direction']} 终点保持 ===",
        f"两步均到达 {report['both_arrived']}/{total}   完成十跳 {report['all_ten']}/{total}"
        f"   失稳 {report['deaths']}   终点偏移超 30 cm {report['endpoint_over_30cm']}",
        f"指令严格到达 {report['strict_arrivals']}/{report['requested']}   "
        f"第 3–8 跳配对用例 {report['paired_through_hop8']}/{total}",
        "",
        "配对指标      单位       热身       终点       变化",
    ]
    paired = report['paired_hops_3_to_8']
    for key, label, unit, scale, digits in (
        ('peak_tilt_deg', '机体倾角峰值', 'deg', 1, 2),
        ('peak_root_offset_m', '水平偏移峰值', 'cm', 100, 2),
        ('peak_xy_speed_m_s', '水平速度峰值', 'm/s', 1, 3),
        ('apex_m', '跳跃顶点高度', 'm', 1, 3),
        ('landing_foot_error_m', '触地水平误差', 'cm', 100, 2),
    ):
        warm, end = (paired[phase][key] for phase in ('warm', 'endpoint'))
        if warm is None or end is None:
            values = ['—'] * 3
        else:
            values = [f'{warm*scale:.{digits}f}', f'{end*scale:.{digits}f}',
                      f'{(end-warm)*scale:+.{digits}f}']
        lines.append(f'{label}  {unit:<4}{values[0]:>9}{values[1]:>11}{values[2]:>11}')
    lines.append(f"数据文件：{path}")
    return '\n' + '\n'.join(lines) + '\n'
