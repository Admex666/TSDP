const is_trotting = (discipline == 'trotting');
const is_greyhound = (discipline == 'greyhound');
const is_greyhound_or_trotting = (discipline == 'greyhound' || discipline == 'trotting');
const is_participant = (entity_type == 'participant');
const is_show_driver_jockey = (is_participant && !is_greyhound);
const min_for_placement = 60;
const min_for_prize = 160;
const min_for_year = 75;

function _race_form_columns(params)
{
    let race_form_stat_columns = [
        {
            formatter: "responsiveCollapse",
            minWidth: 30,
            hozAlign: "center",
            resizable: false,
            headerSort: false,
            responsive: 0,
        }, {
            title: "Date",
            field: "date",
            hozAlign: "right",
            responsive: 1,
            headerSort: false,
            resizable: false,
        }, {
            title: "Track ",
            field: "track",
            hozAlign: "left",
            responsive: 11,
            headerSort: false,
            resizable: false,
        }, {
            title: "First prize",
            field: "first_prize",
            hozAlign: "right",
            responsive: 4,
            headerSort: false,
            resizable: false,
            formatter: function (cell, formatterParams, onRendered)
            {
                formatterParams.symbol = ` ${cell.getData().first_prize_currency}`;
                return this.formatters.money(cell, formatterParams, onRendered);
            },
            formatterParams: {
                decimal: ",",
                thousand: " ",
                symbol: " HUF",
                symbolAfter: true,
                precision: false,
            },
        }, {
            title: "Distance",
            field: "distance",
            hozAlign: "right",
            responsive: 4,
            headerSort: false,
            resizable: false,
            formatter: function(cell, formatterParams, onRendered){
                return cell.getValue() + ' m';
            },
        }, {
            title: "Start",
            field: "start_type",
            hozAlign: "center",
            responsive: 4,
            headerSort: false,
            resizable: false,
            visible: is_trotting,
            formatter: "lookup",
            formatterParams: {'auto': 'A', 'volte': 'F', 'flying': 'R', undefined: ''},
        }, {
            title: "Race",
            field: "race_yearly_num",
            hozAlign: "left",
            responsive: 4,
            headerSort: false,
            maxWidth: 450,
            vertAlign: 'middle',
            formatter: function(cell, formatterParams, onRendered){
                let cell_data = cell.getData();
                let cell_element = cell.getElement();
                cell_element.style.whiteSpace = "normal";
                cell_element.style.lineHeight = "normal";
                
                if(cell_data.race_id == 0){
                    return `<div>${cell_data.race} (${cell_data.yearly})</div>`;        
                }
                return `<div><a href="/races/${discipline}/${cell_data.race_id}">${cell_data.race}</a> (${cell_data.yearly})</div>`;
            },
        }, {
            title: "Surface",
            field: "surface",
            hozAlign: "center",
            responsive: 4,
            headerSort: false,
            resizable: false,
            visible: !is_greyhound_or_trotting,
        }, {
            title: "Placement",
            field: "placement",
            hozAlign: "center",
            responsive: 2,
            headerSort: false,
            resizable: false,
        }, {
            title: "Time",
            field: "km_time",
            hozAlign: "right",
            responsive: 5,
            headerSort: false,
            resizable: false,
            visible: is_trotting,
        }, {
            title: "Prize",
            field: "prize",
            minWidth: min_for_prize,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 3,
            headerSort: false,
            resizable: false,
            formatter:"money",
            formatterParams: {
                decimal: ",",
                thousand: " ",
                symbol: " HUF",
                symbolAfter: true,
                precision: false,
            },
        }, {            
            title: "Jockey",
            field: "driver_jockey",
            hozAlign: "center",
            responsive: 3,
            headerSort: false,
            resizable: false,
            formatter: "link",
            formatterParams: {
                urlField: "driver_jockey_id",
                urlPrefix: "/race_career_driver_jockey/"+discipline+"/",
            },
            visible: is_show_driver_jockey,
        }, {
            title: "Participant",
            field: "participant",
            hozAlign: "center",
            responsive: 3,
            headerSort: false,
            resizable: false,
            formatter: "link",
            formatterParams: {
                urlField: "participant_id",
                urlPrefix: "/race_career_participant/"+discipline+"/",
            },
            visible: !is_participant,
        }, {
            title: "Weight",
            field: "weight",
            hozAlign: "center",
            responsive: 8,
            headerSort: false,
            resizable: false,
            visible: !is_greyhound_or_trotting,
        }, {
            title: "Form",
            field: "form",
            hozAlign: "center",
            responsive: 9,
            headerSort: false,
            visible: !is_greyhound_or_trotting,
        }, {
        title: "Video",
        field: "video",
        hozAlign: "center",
        vertAlign: 'middle',
        responsive: 10,
        headerSort: false,
        formatter: function(cell, formatterParams, onRendered){
            let link = cell.getData().video_link;
            if (link == undefined){
                return '';
            }
            return `<div><a href="${link}" target="_blank"><img src="${yt_logo_url}"></a></div>`;
        },
        resizable: false,
    },
    ].filter(x => x);

    let race_form_stat_columns_greyhound = [
        {
            formatter: "responsiveCollapse",
            minWidth: 30,
            hozAlign: "center",
            resizable: false,
            headerSort: false,
            responsive: 0,
        }, {
            title: "Date",
            field: "date",
            hozAlign: "right",
            responsive: 1,
            headerSort: false,
            resizable: false,
        }, {
            title: "Race",
            field: "race_yearly_num",
            hozAlign: "left",
            responsive: 3,
            headerSort: false,
            maxWidth: 220,
            vertAlign: 'middle',
            formatter: function(cell, formatterParams, onRendered){
                let cell_data = cell.getData();
                let cell_element = cell.getElement();
                cell_element.style.whiteSpace = "normal";
                cell_element.style.lineHeight = "normal";
                
                if(cell_data.race_id == 0){
                    return `<div>${cell_data.race} (${cell_data.yearly})</div>`;        
                }
                return `<div><a href="/races/${discipline}/${cell_data.race_id}">${cell_data.race}</a> (${cell_data.yearly})</div>`;
            },
        }, {
            title: "number",
            field: "number",
            hozAlign: "center",
            responsive: 12,
            headerSort: false,
            resizable: false,
            formatter: function(cell, formatterParams, onRendered){
                let asset_id = cell.getValue();
                if(isNaN(asset_id)){
                    return asset_id;
                }
                return `<img class='greyhound_trap_img' src="${asset_base_url}${discipline}/${asset_id}?ver=${version}" alt="${asset_id}"></img>`;
              },
        }, {
            title: "Placement",
            field: "placement",
            hozAlign: "center",
            responsive: 2,
            headerSort: false,
            resizable: false,
        }, {
            title: "winner name",
            field: "winner_name",
            hozAlign: "left",
            responsive: 4,
            headerSort: false,
            resizable: false,
            maxWidth: 220,
            formatter: "link",
            formatterParams: {
                urlField: "winner_id",
                urlPrefix: "/race_career_participant/"+discipline+"/",
            },
        }, {
            title: "winner time",
            field: "winner_time",
            hozAlign: "center",
            responsive: 5,
            headerSort: false,
            resizable: false,
        }, {
            title: "est time",
            field: "est_time",
            hozAlign: "center",
            responsive: 10,
            headerSort: false,
            resizable: false,
        }, {
            title: "race_grade",
            field: "race_grade",
            headerSort: false,
            hozAlign: "center",
            responsive: 7,
        }, {
            title: "runner_grade",
            field: "runner_grade",
            responsive: 7,
            headerSort: false,
            hozAlign: "center",
            maxWidth: 95,
            headerHozAlign: "center",
            titleFormatter: function(cell, formatterParams, onRendered){
                let cell_element = cell.getElement();
                cell_element.style.whiteSpace = "normal";
                cell_element.style.lineHeight = "normal";
                return cell.getValue();
            },
        }, {
            title: "Prize",
            field: "prize",
            hozAlign: "right",
            responsive: 3,
            headerSort: false,
            resizable: false,
            formatter:"money",
            formatterParams: {
                decimal: ",",
                thousand: " ",
                symbol: " HUF",
                symbolAfter: true,
                precision: false,
            },
        }, {
            title: "Weight",
            field: "weight",
            hozAlign: "center",
            responsive: 4,
            headerSort: false,
            resizable: false,
            formatter: function(cell, formatterParams, onRendered){
                let value = cell.getValue();
                if (value == undefined){
                    return '';
                }
                return parseFloat(cell.getValue()).toFixed(2);
            },
        }, {
            title: "Distance",
            field: "distance",
            hozAlign: "right",
            headerHozAlign: "center",
            responsive: 13,
            headerSort: false,
            resizable: false,
            formatter: function(cell, formatterParams, onRendered){
                return cell.getValue() + ' m';
            },
        }, {
            title: "Video",
            field: "video",
            hozAlign: "center",
            vertAlign: 'middle',
            responsive: 8,
            headerSort: false,
            formatter: function(cell, formatterParams, onRendered){
                let link = cell.getData().video_link;
                if (link == 'null'){
                    return '';
                }
                return `<div><a href="${link}" target="_blank"><img src="${yt_logo_url}"></a></div>`;
            },
            resizable: false,
        }, {
            title: "going",
            field: "going",
            responsive: 8,
            headerSort: false,
            hozAlign: "center",
            formatter: function(cell, formatterParams, onRendered){
                let value = cell.getValue();
                if (value == undefined){
                    return '';
                }
                return dictionary[current_locale].columns.value_map.going[value.toString()];
            },
        }, {
            title: "runners",
            field: "runners",
            responsive: 6,
            headerSort: false,
            hozAlign: "center",
            resizable: false,
        }, {
            title: "Track ",
            field: "track",
            hozAlign: "center",
            maxWidth: 120,
            responsive: 9,
            headerSort: false,
            resizable: false,
            headerHozAlign: "center",
            formatter: function(cell, formatterParams, onRendered){
                let cell_element = cell.getElement();
                cell_element.style.whiteSpace = "normal";
                cell_element.style.lineHeight = "normal";
                return cell.getValue();
            },
        }, 
    ].filter(x => x);

    return is_greyhound ? race_form_stat_columns_greyhound: race_form_stat_columns;
}

function _by_age_columns(param)
{
    let by_age_columns = [
        {
            formatter: "responsiveCollapse",
            minWidth: 30,
            hozAlign: "center",
            resizable: false,
            headerSort: false,
            responsive: 0,
        }, {
            title: "Age",
            field: "age",
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 1,
            headerSort: false,
            formatter: function(cell, formatterParams, onRendered){
                formatterParams = dictionary[current_locale]['columns']['value_map']['age'];
                return this.formatters.lookup(cell, formatterParams, onRendered);
            },
            resizable: false,
        }, {
            title: "Race ",
            field: "race_count",
            minWidth: 80,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 3,
            headerSort: false,
            resizable: false,
        }, {
            title: "1st",
            field: "1st",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 4,
            headerSort: false,
            resizable: false,
        }, {
            title: "2nd",
            field: "2nd",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 5,
            headerSort: false,
            resizable: false,
        }, {
            title: "3rd",
            field: "3rd",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 6,
            headerSort: false,
            resizable: false,
        }, {
            title: "4th",
            field: "4th",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 7,
            headerSort: false,
            visible: is_greyhound,
            resizable: false,
        }, {
            title: "5th",
            field: "5th",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 8,
            headerSort: false,
            visible: is_greyhound,
            resizable: false,
        }, {
            title: "4th - 5th",
            field: "4th_5th",
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 9,
            headerSort: false,
            visible: !is_greyhound,
            resizable: false,
        }, {
            title: "6th",
            field: "6th",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 10,
            headerSort: false,
            visible: is_greyhound,
            resizable: false,
        }, {
            title: "Total Prize",
            field: "total_prize",
            minWidth: min_for_prize,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 2,
            formatter:"money",
            formatterParams: {
                decimal: ",",
                thousand: " ",
                symbol: " HUF",
                symbolAfter: true,
                precision: false,
            },
            headerSort: false,
            
        },
    ].filter(x => x);

    return by_age_columns
}

function total_results_paramLookup(cell){

    cell_val = cell.getValue();

    let ret = {'total_results': capitalize(dictionary[current_locale]['general']['total'])};
    if(cell_val != 'total_results'){
        /*
        ret[cell_val] = '<a href="#">' + cell_val + '</a>';
        */
       ret[cell_val] = cell_val;
    }

    return ret;
}

function monthFormatter(cell, formatterParams, onRendered) {
    month_int = parseInt(cell.getValue());
    if (Number.isNaN(month_int)){
        if(cell.getValue() == 'total_results'){
            return capitalize(dictionary[current_locale]['general']['total']);
        }
      return cell.getValue();
    }
    let objDate = new Date();
    objDate.setDate(1);
    objDate.setMonth(month_int-1);
    let month_str = objDate.toLocaleString(current_locale, {month: "long"});
    return capitalize(month_str);
}

function totalFormatter(cell, formatterParams, onRendered){
    if(cell.getValue() == 'total_results'){
        return capitalize(dictionary[current_locale]['general']['total']);
    }

    return cell.getValue();
}

function race_careerLinkFormatter(cell, formatterParams, onRendered){
    if(cell.getValue() == 'total_results'){
        return capitalize(dictionary[current_locale]['general']['total']);
    }
    
    let id = cell.getData()['id'];
    return '<a href="/race_career_'+formatterParams.grouping+'/'+discipline+'/'+ id +'">' + cell.getValue() + '</a>';
}

function _participant_years(param){
    let participant_years_columns = [
        {
            formatter: "responsiveCollapse",
            minWidth: 30,
            hozAlign: "center",
            resizable: false,
            headerSort: false,
            responsive: 0,
        }, {
            title: "Year",
            field: "year",
            minWidth: min_for_year,
            hozAlign: "left",
            responsive: 1,
            headerSort: false,
            formatter: "lookup",
            formatterParams: total_results_paramLookup,
        }, {
            title: "Runs",
            field: "runs",
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 3,
            headerSort: false,
        }, {
            title: "1st",
            field: "1st",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 4,
            headerSort: false,
        }, {
            title: "2nd",
            field: "2nd",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 5,
            headerSort: false,
        }, {
            title: "3rd",
            field: "3rd",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 6,
            resizable: false,
            headerSort: false,
        }, {
            title: "4th",
            field: "4th",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 7,
            headerSort: false,
            visible: is_greyhound,
            resizable: false,
        }, {
            title: "5th",
            field: "5th",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 8,
            visible: is_greyhound,
            resizable: false,
            headerSort: false,
        }, {
            title: "4th - 5th",
            field: "4th_5th",
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 9,
            visible: !is_greyhound,
            resizable: false,
            headerSort: false,
        }, {
            title: "6th",
            field: "6th",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 10,
            visible: is_greyhound,
            resizable: false,
            headerSort: false,
        }, {
            title: "Unplaced",
            field: "unplaced",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 11,
            visible: !is_greyhound,
            headerSort: false,
        }, {
            title: "Total Prize",
            field: "total_prize",
            minWidth: min_for_prize,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 2,
            formatter:"money",
            formatterParams: {
                decimal: ",",
                thousand: " ",
                symbol: " HUF",
                symbolAfter: true,
                precision: false,
            },
            headerSort: false,
        },
    ].filter(x => x);

    return participant_years_columns
}

function _annual_stat(param){

    const grouping = param.url.params[param.url.params.length-2];
    const is_monthly = (grouping == 'monthly');
    let grouping_table_header = param.url.params_dict.grouping_table_header;
    const has_race_career = !(['monthly', 'distance'].includes(grouping))

    let lowPlaceColumns =
    discipline == "greyhound" ?
      [
        {
          title: "4th",
          field: "4th",
          hozAlign: "right",
          headerHozAlign: "right",
          responsive: 8,
          headerSort: false,
        }, {
          title: "5th",
          field: "5th",
          hozAlign: "right",
          headerHozAlign: "right",
          responsive: 9,
          headerSort: false,
        }, {
          title: "6th",
          field: "6th",
          hozAlign: "right",
          headerHozAlign: "right",
          responsive: 10,
          headerSort: false,
        }
      ] :
      [
        {
          title: "4th - 5th",
          field: "4th_5th",
          hozAlign: "right",
          headerHozAlign: "right",
          responsive: 8,
          headerSort: false,
        }
      ];

    let columns = [
        {
            formatter: "responsiveCollapse",
            width: 30,
            minWidth: 30,
            hozAlign: "center",
            resizable: false,
            responsive: 0,
        }, {
            title: grouping_table_header.charAt(0).toUpperCase() + grouping_table_header.slice(1),
            field: grouping_table_header,
            hozAlign: "left",
            formatterParams: {
                grouping: grouping
            },
            formatter: is_monthly ? monthFormatter : has_race_career? race_careerLinkFormatter : totalFormatter,
            responsive: 1,
            headerSort: false,
        }, {
            title: 'Call name',
            field: 'short_name',
            hozAlign: "left",
            formatterParams: {
                grouping: grouping
            },
            formatter: is_monthly ? monthFormatter : has_race_career? race_careerLinkFormatter : totalFormatter,
            visible: is_greyhound && grouping_table_header == 'participant',
            responsive: 9,
            headerSort: false,
        }, {
            title: "Runs",
            field: "runs",
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 3,
            headerSort: false,
            formatter: function(cell, formatterParams, onRendered){
                if(cell.getValue() == '0'){
                    return cell.getValue();
                }
                let position = cell.getRow().getPosition();
                return `<a href="#" class="filtered-race-forms" onclick="return false;" position="${position}">${cell.getValue()}</a>`;
            },
        }, {
            title: "1st",
            field: "1st",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 4,
            headerSort: false,
        }, {
            title: "1st %",
            field: "1st_percentage",
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 5,
            headerSort: false,
        }, {
            title: "2nd",
            field: "2nd",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 6,
            headerSort: false,
        }, {
            title: "3rd",
            field: "3rd",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 7,
            headerSort: false,
        }, 
        ...lowPlaceColumns,
        {
            title: "Total Prize",
            field: "total_prize",
            minWidth: min_for_prize,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 2,
            formatter:"money",
            formatterParams: {
                decimal: ",",
                thousand: " ",
                symbol: " HUF",
                symbolAfter: true,
                precision: false,
            },
            headerSort: false,
        },
    ].filter(x => x);

    return columns;
}

function _progeny_stat(param){

    let grouping_table_header = 'participant';

    let lowPlaceColumns =
    discipline == "greyhound" ?
      [
        {
          title: "4th",
          field: "4th",
          hozAlign: "right",
          headerHozAlign: "right",
          responsive: 8,
          headerSort: false,
        }, {
          title: "5th",
          field: "5th",
          hozAlign: "right",
          headerHozAlign: "right",
          responsive: 9,
          headerSort: false,
        }, {
          title: "6th",
          field: "6th",
          hozAlign: "right",
          headerHozAlign: "right",
          responsive: 10,
          headerSort: false,
        }
      ] :
      [
        {
          title: "4th - 5th",
          field: "4th_5th",
          hozAlign: "right",
          headerHozAlign: "right",
          responsive: 8,
          headerSort: false,
        }
      ];

    let columns = [
        {
            formatter: "responsiveCollapse",
            width: 30,
            minWidth: 30,
            hozAlign: "center",
            resizable: false,
            responsive: 0,
        }, {
            title: 'born',
            field: 'born',
            hozAlign: "left",
            visible: grouping_table_header == 'participant',
            responsive: 4,
            headerSort: false,
        }, {
            title: 'Call name',
            field: 'short_name',
            hozAlign: "left",
            visible: is_greyhound && grouping_table_header == 'participant',
            responsive: 2,
            headerSort: false,
        }, {
            title: 'progeny_name',
            field: 'progeny_name',
            hozAlign: "left",
            visible: grouping_table_header == 'participant',
            formatter: "link",
            formatterParams: {
                urlField: "id",
                urlPrefix: "/race_career_participant/"+discipline+"/",
            },
            responsive: 1,
            headerSort: false,
        }, {
            title: 'Other Parent Name',
            field: 'other_parent_name',
            hozAlign: "left",
            visible: grouping_table_header == 'participant',
            formatter: "link",
            formatterParams: {
                urlField: "other_parent_id",
                urlPrefix: "/race_career_participant/"+discipline+"/",
            },
            responsive: 3,
            headerSort: false,
        }, {
            title: "Runs",
            field: "race_count",
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 4,
            headerSort: false,
        }, {
            title: "1st",
            field: "1st",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 5,
            headerSort: false,
        }, {
            title: "2nd",
            field: "2nd",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 6,
            headerSort: false,
        }, {
            title: "3rd",
            field: "3rd",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 7,
            headerSort: false,
        },
        ...lowPlaceColumns,
        {
            title: "Unplaced",
            field: "unplaced",
            minWidth: min_for_placement,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 11,
            visible: !is_greyhound,
            headerSort: false,
        }, {
            title: "Total Prize",
            field: "total_prize",
            minWidth: min_for_prize,
            hozAlign: "right",
            headerHozAlign: "right",
            responsive: 12,
            formatter:"money",
            formatterParams: {
                decimal: ",",
                thousand: " ",
                symbol: " HUF",
                symbolAfter: true,
                precision: false,
            },
            headerSort: false,
        },
    ].filter(x => x);

    return columns;
}

let race_career_tables = {
    'by_age_stat': _by_age_columns,
    'race_form_stat': _race_form_columns,
    'years_stat': _participant_years,
    'annual_stat': _annual_stat,
    'progeny_stat': _progeny_stat
}
