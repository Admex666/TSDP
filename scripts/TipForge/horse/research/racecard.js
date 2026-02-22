var wagercountry = "";
var wagercurrency = "Ft";
var wagercurrencyrate = 1;
var wagerraceid = 0;
var wagertype = "";
var rankedwager = false;
var stakeamount = 0;
var leg1 = false;
var leg2 = false;
var leg3 = false;
var leg4 = false;
var leg5 = false;
var legR = false;
//var legK = false;
var fra_runners = [];
var fra_rrunners = [];
var fra_replhorse = 0;
var fra_riskvalue = 4;
var fra_minhorses = 0;
var fra_hasspot = false;
var fra_xcount = 0;
var minstake = 0;
var maxstake = 0;
var wagerUpdateTimer;

function resetGlobalWagerVars() {
	wagercountry = "";
	wagercurrency = "Ft";
	wagercurrencyrate = 1;
	wagerraceid = 0;
	wagertype = "";
	rankedwager = false;
	stakeamount = 0;
	leg1 = false;
	leg2 = false;
	leg3 = false;
	leg4 = false;
	leg5 = false;
	legR = false;
	//legK = false;
	fra_runners = [];
	fra_rrunners = [];
	fra_replhorse = 0;
	fra_riskvalue = 4;
	fra_minhorses = 0;
	fra_hasspot = false;
	fra_xcount = 0;
	minstake = 0;
	maxstake = 0;
}

function initWager() {

	wagertype = "";

	$("#racecardWager").data("pooldata_TET", { horses: 1, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: false, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_HEL", { horses: 1, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: false, flexi25: false, flexi10: false } );

	$("#racecardWager").data("pooldata_Simple Gagnant", { horses: 1, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: false, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Simple Place", { horses: 1, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: false, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Simple Gagnant Jackpot", { horses: 1, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: false, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Simple Place Jackpot", { horses: 1, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: false, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_E Simple Gagnant", { horses: 1, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: false, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_E Simple Place", { horses: 1, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: false, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Simple Gagnant International", { horses: 1, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: false, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Simple Place International", { horses: 1, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: false, flexi25: false, flexi10: false } );

	$("#racecardWager").data("pooldata_BEF", { leg1: true, leg2: true, leg3: false, leg4: false, leg5: false, legK: false, legR: false, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_HBE", { leg1: true, leg2: true, leg3: true, leg4: false, leg5: false, legK: false, legR: false, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_BEF2", { horses: 2, hasboxed: true, hasx: true, hasall: true, hasrepl: true, hasspot: false, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_HBE2", { horses: 3, hasboxed: true, hasx: true, hasall: true, hasrepl: true, hasspot: false, flexi50: true, flexi25: false, flexi10: false } );

	$("#racecardWager").data("pooldata_Couple Gagnant", { horses: 2, hasboxed: false, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Couple Place", { horses: 2, hasboxed: false, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Couple Ordre", { horses: 2, hasboxed: true, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_E Couple Gagnant", { horses: 2, hasboxed: false, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_E Couple Place", { horses: 2, hasboxed: false, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_E Couple Ordre", { horses: 2, hasboxed: true, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Couple Ordre International", { horses: 2, hasboxed: true, hasx: true, hasall: true, hasrepl: false, hasspot: false, flexi50: false, flexi25: false, flexi10: false } );

	$("#racecardWager").data("pooldata_Couple Ordre Ranked", { leg1: true, leg2: true, leg3: false, leg4: false, leg5: false, legK: false, legR: true, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Couple Ordre International Ranked", { leg1: true, leg2: true, leg3: false, leg4: false, leg5: false, legK: true, legR: false, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: false, flexi25: false, flexi10: false } );

	$("#racecardWager").data("pooldata_Trio", { horses: 3, hasboxed: false, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Trio Ordre", { horses: 3, hasboxed: true, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_E Trio", { horses: 3, hasboxed: false, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_E Trio Ordre", { horses: 3, hasboxed: true, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Trio Ordre International", { horses: 3, hasboxed: true, hasx: true, hasall: true, hasrepl: false, hasspot: false, flexi50: false, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Tierce", { horses: 3, hasboxed: true, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Classic Tierce", { horses: 3, hasboxed: true, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: false, flexi10: false } );

	$("#racecardWager").data("pooldata_Trio Ordre Ranked", { leg1: true, leg2: true, leg3: true, leg4: false, leg5: false, legK: false, legR: true, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Trio Ordre International Ranked", { leg1: true, leg2: true, leg3: true, leg4: false, leg5: false, legK: false, legR: true, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: false, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Tierce Ranked", { leg1: true, leg2: true, leg3: true, leg4: false, leg5: false, legK: false, legR: true, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: true, flexi25: false, flexi10: false } );

	$("#racecardWager").data("pooldata_Multi", { horses: 4, hasboxed: false, hasx: true, hasall: true, hasrepl: false, hasspot: true, flexi50: true, flexi25: true, flexi10: false} );
	$("#racecardWager").data("pooldata_Mini Multi", { horses: 4, hasboxed: false, hasx: true, hasall: true, hasrepl: false, hasspot: true, flexi50: true, flexi25: true, flexi10: false } );
	$("#racecardWager").data("pooldata_Deux Sur Quatre", { horses: 2, hasboxed: false, hasx: true, hasall: false, hasrepl: true, hasspot: true, flexi50: true, flexi25: true, flexi10: false } );

	$("#racecardWager").data("pooldata_Quarte+", { horses: 4, hasboxed: true, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Super4", { horses: 4, hasboxed: true, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: false, flexi10: false } );

	$("#racecardWager").data("pooldata_Super4 Ranked", { leg1: true, leg2: true, leg3: true, leg4: true, leg5: false, legK: false, legR: true, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_Quarte+ Ranked", { leg1: true, leg2: true, leg3: true, leg4: true, leg5: false, legK: false, legR: true, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: true, flexi25: false, flexi10: false } );

	$("#racecardWager").data("pooldata_Quinte+", { horses: 5, hasboxed: true, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: true, flexi10: false } );

	$("#racecardWager").data("pooldata_Quinte+ Ranked", { leg1: true, leg2: true, leg3: true, leg4: true, leg5: true, legK: false, legR: true, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: true, flexi25: true, flexi10: false } );
	$("#racecardWager").data("pooldata_OTO", { leg1: true, leg2: true, leg3: true, leg4: true, leg5: true, legK: false, legR: false, hasboxed: false, hasx: false, hasall: false, hasrepl: false, hasspot: false, flexi50: true, flexi25: false, flexi10: false } );
	$("#racecardWager").data("pooldata_OTO2", { horses: 5, hasboxed: true, hasx: true, hasall: true, hasrepl: false, hasspot: false, flexi50: true, flexi25: false, flexi10: false } );

	$("#racecardWager").data("pooldata_NP5", { horses: 5, hasboxed: false, hasx: true, hasall: true, hasrepl: true, hasspot: true, flexi50: true, flexi25: true, flexi10: false } );

	$("#racecardWager .betTypeList li.bettypes").click(function (event) {
		if($(this).hasClass("active")) {
			return false;
		}
		var newwagertype = $(this).data("wagertype");
		setWagerType(newwagertype);
		//setBooster($(this));
		$(this).addClass("active");
		$(this).siblings(".active").removeClass("active");
		$("#racecardWager .wagertypehelp.hidden").removeClass("hidden");
		$("#racecardWager .betTypeInfo").html("");
		$.get("/racecard/topdividends?pool="+encodeURIComponent(newwagertype), function(data) {
			$(".topdividends").html(data);
		});

		return false;
	});

	$("#racecardWager .selectedValueWrapper").click(function () {
		$("#racecardWager .selectDropdown").toggleClass("active");
	});

	$("#racecardWager .stakeselect").click(function(stakevent) {
		if($(stakevent.target).is("li")) {
			setStakeAmount($(stakevent.target).data("amount"), $(stakevent.target).html());
		}
	});

	$("#racecardWager .deleteSelection").click(function(event) {
		if(wagertype != "") {
			resetWager(false);
		}
	});

	$("#racecardWager .showCombinations").click(function(event) {
		if($("#racecardWager .racecardCombinations").hasClass("active")) {
			$("#racecardWager .racecardCombinations").removeClass("active");
			$("#racecardWager .racecardCombinations .scrollWrapper").html("");
		}
		else {
			$("#racecardWager .racecardCombinations").addClass("active");
			showCombinations(getWagerPostVars(getSelectedRunners()));
		}
	});

	$("#racecardWager .racecardCombinations .close").click(function(event) {
		$("#racecardWager .racecardCombinations.active").removeClass("active");
		$("#racecardWager .racecardCombinations .scrollWrapper").html("");
		return false;
	});

	$("#racecardWager .wagersendbutton").click(function(event) {
		var el = this;
		if($(el).hasClass("disabled")) {
			return false;
		}

		$("#racecardWager .bet-summary-box").addClass("step2");
		$("#racecardWager").addClass("overlayActive");
		$("#overlay").addClass("active");

		if((wagertype == "Simple Gagnant" || wagertype == "Simple Place") && fra_runners.length == 1 && stakeamount <= 720000) {
			var jackpotwagertype = "";
			if(wagertype == "Simple Gagnant") {
				jackpotwagertype = "Simple Gagnant Jackpot";
			}
			else if(wagertype == "Simple Place") {
				jackpotwagertype = "Simple Place Jackpot";
			}
			if($("#racecardWager li.bt_"+jackpotwagertype.replace(/\W/g, "")).length) {

				var jackpotextra = parseInt(((stakeamount/2)/100), 10);
				$("#racecardWager .jackpotConfirm .jackpotFee").text(jackpotextra);
				$("#racecardWager .racecardSubmitConfirm").html($("#racecardWager .jackpotConfirm").html());

				$("#racecardWager .racecardSubmitConfirm .jackpotcancelbutton").click(function(event) {
					placeBet(false);
				});

				$("#racecardWager .racecardSubmitConfirm .jackpotconfirmbutton").click(function(event) {
					var oldstakeamount = stakeamount;
					setWagerTypeComplete(jackpotwagertype, false);
					var jackpotextra = parseInt((oldstakeamount/2), 10);
					var jackpotstake = (oldstakeamount + jackpotextra);
					if(jackpotstake != stakeamount) {
						setStakeAmount(jackpotstake, "");
					}
					$("#racecardWager li.bt_"+jackpotwagertype.replace(/\W/g, "")).addClass("active").siblings(".active").removeClass("active");
					placeBet(false);
				});
				return;
			}
		}

		placeBet(false);
	});


	$("#racecardWager .bettypetoggler").click(function() {
		$("#racecardWager .betTypeList li.pooltoggle").toggleClass("hidden");
		$("#racecardWager .racecardOptions").toggleClass("hidden");
		$("#racecardWager .bettypetoggler").toggleClass("hidden");
		resetWager(true);
	});

	$("#racecardWager .rankedtoggler").click(function() {
		$("#racecardWager .betTypeList li.rankedhidden").toggleClass("hidden");
		$("#racecardWager .rankedtoggler").toggleClass("hidden");
		resetWager(true);
	});

	$("#racecardWager .wagertypehelp").click(function() {
		var postvars = "pool="+encodeURIComponent(wagertype)+"&country="+wagercountry;
		if($("#racecardWager .betTypeInfo").html() == "") {
			$.post("/betcalculator/poolinfo", postvars, function(data) {
				$("#racecardWager .betTypeInfo").html(data);
			});
		}
		else {
			$("#racecardWager .betTypeInfo").html("");
		}
	});

	initWagerFra(true);
}

function initWagerSimple() {
	$("#racecard .bets .checkbox").click(function (event) {
		var el = this;
		if($(el).hasClass("disabled")) {
			return false;
		}

		var leg = $(el).data("leg");
		if(leg == "R") {
			$("#racecard .bets .checkbox.legR.checked").each(function() {
				if($(this).parent().data("startnr") != $(el).parent().data("startnr")) {
					$(this).removeClass("checked");
				}
			});
		}

		$(el).toggleClass("checked");
		//var checked = $(el).hasClass("checked");

		//clickSimpleWagerRunner(el, leg, checked);
		//checkAllLegsState();
		showSelection(true);
	});

//	$("#racecard .racecardListHeader .allleg").click(function (event) {
//		var el = this;
//		if($(el).hasClass("disabled"))
//			return false;
//
//		var checked = $(el).hasClass("selected");
//		var leg = $(el).data("leg");
//
//		$("#racecard .bets .checkbox.leg"+leg).each(function () {
//			if(!$(this).hasClass("disabled")) {
//				var runnerchecked = $(this).hasClass("checked");
//				if(runnerchecked == checked) {
//					$(this).toggleClass("checked");
//					clickSimpleWagerRunner(this, leg, !checked);
//				}
//			}
//		});
//
//		checkAllLegsState();
//		showSelection(true);
//	});
}

function initWagerFra(initoptions) {
	$("#racecard .bets .checkbox").click(function (event) {
		var el = this;
		if($(el).hasClass("disabled")) {
			return false;
		}

		var startnr = $(el).parent().data("startnr");
		//Ist Ersatzpferd Button aktiviert?
		if($("#WagerReplacementButton").hasClass("active")) {

			if(fra_replhorse == startnr) {
				fra_replhorse = 0;
				$("#racecard .racecardList .cell-t").html("");
			}
			else {
				var fra_runnersindex = $.inArray(startnr, fra_runners);
				if(fra_runnersindex != -1) {
					return false;
				}
				fra_replhorse = startnr;
				$("#racecard .racecardList .cell-t").html("");
				$(el).siblings(".cell-t").html("T");
			}
			$("#WagerReplacementButton").removeClass("active");
		}
		else {
			var checked = $(el).hasClass("checked");
			clickFraWagerRunner(startnr, !checked, el);
			$(el).toggleClass("checked");
		}
		showSelection(true);
	});

	if(initoptions) {
		$("#WagerReplacementButton").click(function(event) {
			if($(this).hasClass("disabled"))
				return false;
			$(this).toggleClass("active");
		});

		$("#WagerBoxedButton").click(function(event) {
			if($(this).hasClass("disabled")) {
				return false;
			}
			$(this).toggleClass("active");
			showSelection(true);
			//countCombinations(null);
		});

		$("#WagerXButton").click(function(event) {
			if($(this).hasClass("disabled"))
				return false;
			fra_runners.push("X");
			fra_xcount++;
			showSelection(true);
		});

		$("#WagerSpotButton").click(function(event) {
			if($(this).hasClass("disabled"))
				return false;

			fra_runners.push("S");
			fra_hasspot = true;
			showSelection(true);
		});

		$("#WagerAllButton").click(function(event) {
			if($(this).hasClass("disabled"))
				return false;

			$("#racecard .bets .checkbox:not(.checked)").each(function() {
				var startnr = $(this).parent().data("startnr");
				clickFraWagerRunner(startnr, true, this);
				$(this).toggleClass("checked");
			});

			showSelection(true);
		});

		$("#racecardWager .flexiselect span.check").click(function(event) {
			if($(this).hasClass("disabled")) {
				return false;
			}
			if(!$(this).hasClass("active")) {
				var fleximode = $(this).parent().data("flexi");
				var flexiamount = (minstake/(100/parseInt(fleximode,10)));
				setStakeAmount(flexiamount, "");
				$("#racecardWager .flexiselect span.check.active").removeClass("active");
				$(this).addClass("active");
			}
			else {
				setFirstStakeAmount();
				$(this).removeClass("active");
			}
		});

		$("#racecardWager .multiselect span.check").click(function(event) {
			if($(this).hasClass("disabled"))
				return false;
			if($(this).hasClass("active"))
				return false;
			$("#racecardWager .multiselect span.check.active").removeClass("active");
			$(this).addClass("active");
			var multimode = $(this).parent().data("multi");
			fra_riskvalue = multimode;
			countCombinations(null);
		});

		$("#racecardWager .betcountselect span.check").click(function(event) {
			if($(this).hasClass("disabled")) {
				return false;
			}
			if($(this).hasClass("active")) {
				return false;
			}
			$("#racecardWager .betcountselect span.check.active").removeClass("active");
			$(this).addClass("active");
			countWagerAmount();
		});
	}
}

//function clickSimpleWagerRunner(el, leg, checked) {
//
//	return; //da es keine K spalte gibt, nicht noetig
//
//	if(!checked) {
//		if(leg == 0) {
//			if(leg1) {
//				$(el).siblings(".leg1.disabled").removeClass("disabled");
//			}
//			if(leg2) {
//				$(el).siblings(".leg2.disabled").removeClass("disabled");
//			}
//			if(leg3) {
//				$(el).siblings(".leg3.disabled").removeClass("disabled");
//			}
//			if(leg4) {
//				$(el).siblings(".leg4.disabled").removeClass("disabled");
//			}
//			if(leg5) {
//				$(el).siblings(".leg4.disabled").removeClass("disabled");
//			}
//	    }
//		else if(legK && $(el).siblings(".checked").length == 0) {
//			$(el).siblings(".leg0.disabled").removeClass("disabled");
//		}
//	}
//	else {
//		if(leg == 0)
//			$(el).siblings(":not(.disabled)").addClass("disabled");
//		else
//			$(el).siblings(".leg0:not(.disabled)").addClass("disabled");
//	}
//}

function clickFraWagerRunner(startnr, state, checkbox) {
	var fra_runnersindex = $.inArray(startnr, fra_runners);
	var fra_rrunnersindex = $.inArray(startnr, fra_rrunners);
	if(state) {
		if(fra_xcount > 0) {
			if(fra_runners.length >= fra_minhorses) {
				if(fra_rrunnersindex == -1) {
					fra_rrunners.push(startnr);
					if(!$(checkbox).hasClass("rhorse")) {
						$(checkbox).addClass("rhorse");
					}
				}
			}
			else {
				if(fra_runnersindex == -1) {
					fra_runners.push(startnr);
					if(startnr == fra_replhorse) {
						fra_replhorse = 0;
						$("#racecard .racecardList .cell-t").html("");
					}
				}
			}
		}
		else {
			if(fra_runnersindex == -1) {
				fra_runners.push(startnr);
				if(startnr == fra_replhorse) {
					fra_replhorse = 0;
					$("#racecard .racecardList .cell-t").html("");
				}
			}
		}
	}
	else {
		if(fra_rrunnersindex != -1) {
			fra_rrunners.splice(fra_rrunnersindex, 1);
			if($(checkbox).hasClass("rhorse")) {
				$(checkbox).removeClass("rhorse");
			}
		}
		else if(fra_xcount == 0 || fra_rrunners.length == 0) {
			if(fra_runnersindex != -1) {
				fra_runners.splice(fra_runnersindex, 1);
			}
		}
		else {
			return false;
		}
	}
	return true;
}

//function checkAllLegsState() {
//	for (var i = 0; i <= 5; i++) {
//		var opencheckboxes = $("#racecard .bets .checkbox.leg"+i).not(".disabled").not(".checked").length;
//		var checkedcheckboxes = $("#racecard .bets .checkbox.leg"+i+".checked").length;
//		if(opencheckboxes > 0) {
//			$("#racecard .racecardListHeader .leg"+i+".selected").removeClass("selected");
//			$("#racecard .racecardListHeader .leg"+i+".disabled").removeClass("disabled");
//		}
//		else if(checkedcheckboxes > 0){
//			$("#racecard .racecardListHeader .leg"+i).not(".disabled").not(".selected").addClass("selected");
//		}
//		else {
//			$("#racecard .racecardListHeader .leg"+i+":not(.disabled)").addClass("disabled");
//		}
//	}
//}

function setWagerType(newwagertype) {
	if(newwagertype != wagertype) {
		if(isWagertypeRanked(newwagertype) && !rankedwager) {
			$.get("/racecard/simplewager?ranked=1&id="+wagerraceid, function(data) {
				rankedwager = true;
				$("#racecardInnerWrapper").html(data);
				setWagerTypeComplete(newwagertype, true);
				initWagerSimple();
				initRunners();
			});
		}
		else if(!isWagertypeRanked(newwagertype) && rankedwager) {
			$.get("/racecard/simplewager?id="+wagerraceid, function(data) {
				rankedwager = false;
				$("#racecardInnerWrapper").html(data);
				setWagerTypeComplete(newwagertype, true);
				initWagerFra(false);
				initRunners();
			});
		}
		else {
			setWagerTypeComplete(newwagertype, true);
		}
	}
	return false;
}

// function setBooster(bettype){
// 	var boosterLabel = $('.racecard-box .racecard-box__booster');
//
// 	boosterLabel.addClass('hidden');
// 	boosterLabel.find('.ph').html('');
//
// 	var booster = bettype.find('.booster');
// 	if(booster.length > 0){
// 		boosterLabel.find('.ph').html(booster.html())
// 		boosterLabel.removeClass('hidden');
// 	}
//
//
// }

function setWagerTypeComplete(newwagertype, resetwager) {
	$("#racecardWager .stakeselect").html($("#Stakes_"+(newwagertype.replace(/\W/g, ""))).html());
	wagertype = newwagertype;
	var stakeconfig = $("#racecardWager").data("stakedata_"+wagertype);
	if(stakeconfig != undefined) {
		minstake = stakeconfig.minstake;
		maxstake = stakeconfig.maxstake;
	}
	setFirstStakeAmount();

	if(resetwager) {
		resetWager(false);
	}

	if(wagertype == "Multi" || wagertype == "Mini Multi") {
		$("#racecardWager .betMultiOptions.hidden").removeClass("hidden");
		if(wagertype == "Multi") {
			$("#racecardWager .betMultiOptions .multiselect7.hidden").removeClass("hidden");
		}
		else {
			$("#racecardWager .betMultiOptions .multiselect7:not(.hidden)").addClass("hidden");
		}
	}
	else {
		$("#racecardWager .betMultiOptions:not(.hidden)").addClass("hidden");
	}

	if(wagertype == "Simple Gagnant Jackpot" || wagertype == "Simple Place Jackpot") {
		$("#racecardWager .nonjackpotbettype:not(.hidden)").addClass("hidden");
		$("#racecardWager .jackpotbettype.hidden").removeClass("hidden");
	}
	else {
		$("#racecardWager .jackpotbettype:not(.hidden)").addClass("hidden");
		$("#racecardWager .nonjackpotbettype.hidden").removeClass("hidden");
	}
}

function showSelection(docount) {

	var legs = getSelectedRunners();
	var legstring = "";
	if(!rankedwager) {
		legstring = "<span>";
		if($("#WagerBoxedButton.active").length) {
			legstring += "BOX ";
		}
		legstring += legs[1]+"</span>";
	}
	else {
		for (leg in legs) {
			if(leg == "R") {
				legstring += "Tartalék ló: ";
			}
			else {
				legstring += leg+". helyre: ";
			}

			legstring += "<span>"+legs[leg]+"</span> ";
		}
	}
	$("#racecardWager .betString .value").html(legstring);

	if(docount) {
		countCombinations(legs);
	}
}

function checkForCountCombinations() {
	if(!rankedwager) {
		return (fra_runners.length > 0 ) ? true : false;
	}
	else {
		return ($("#racecard .bets .checkbox.checked").length > 0) ? true : false;
	}
}

function showCombinations(postvars) {
	$("#racecardWager .racecardCombinations .scrollWrapper").html("");
	$("#racecardWager .racecardCombinations").addClass("spinneractive");
	$.post("/racecard/showcombinations", postvars, function(data) {
		$("#racecardWager .racecardCombinations").removeClass("spinneractive");
		$("#racecardWager .racecardCombinations .scrollWrapper").html(data);

		var li = $("#racecardWager .racecardCombinations .scrollWrapper ul li");
		if(li.length > 0){
			$("#racecardWager .racecardCombinations").addClass('active');
		}
		if(li.length > 9){
			$('.racecardCombinations .scrollWrapper').slimScroll({
				height: 'auto',
				railVisible: true,
				alwaysVisible: true
			});
		}

	});
}

function countCombinations(legs) {
	if(checkForCountCombinations()) {
		if(!legs) {
			legs = getSelectedRunners();
		}
		var postvars = getWagerPostVars(legs);
		$.post("/racecard/countcombinations", postvars, function(data) {

			$("#racecardWager .combinationCount .value").text(data);

			if(data == "0") {
				$("#racecardWager .racecardCombinations .scrollWrapper").html("");
			}
			else {
				showCombinations(postvars);
			}

			//if(!rankedwager) {
				enableDisableOptions(wagertype);
			//}
			countWagerAmount();
		});
	}
	else {
		$("#racecardWager .betStake .value").html("0,00 "+wagercurrency);
		$("#racecardWager .combinationCount .value").text("0");
		$("#racecardWager .racecardCombinations .scrollWrapper").html("");
		switchCssClass($("#racecardWager .wagersendbutton"), "disabled", true);
		//if(!rankedwager) {
			enableDisableOptions(wagertype);
		//}
	}
}

function getSelectedRunners() {
	var legs = [];
	if(!rankedwager) {
		legs[1] = getFraLegString();
	}
	else {
		var legrunners = [];
		var legname = "";
		for (var i = 0; i <= 5; i++) {
			legrunners = [];
			//angeklickte Starter Kommasepariert fuer das entsprechende Leg setzen
			$("#racecard .bets .checkbox.leg"+i+".checked").each(function() {
				var startnr = $(this).parent().data("startnr");
				if($.inArray(startnr, legrunners) == -1) {
					legrunners.push(startnr);
				}
			});
			if(legrunners.length > 0) {
//				if(i == 0)
//					legs["K"] = legrunners.join(",");
//				else
					legs[i] = legrunners.join(",");
			}
		}
		if(legR) {
			legrunners = [];
			$("#racecard .bets .checkbox.legR.checked").each(function() {
				var startnr = $(this).parent().data("startnr");
				if($.inArray(startnr, legrunners) == -1) {
					legrunners.push(startnr);
				}
			});
			if(legrunners.length > 0) {
				legs["R"] = legrunners.join(",");
			}
		}
	}
	//console.log(legs);
	return legs;
}

function getWagerPostVars(legs) {
	//Wettart
	var postvars = "pool="+encodeURIComponent(wagertype)+"&raceid="+wagerraceid;
	var legname = "";
	for (leg in legs) {
		legname = "leg"+leg;
		postvars += "&"+legname+"="+legs[leg];
	}
	if(!rankedwager) {
		if(!$("#WagerBoxedButton").hasClass("disabled") && $("#WagerBoxedButton").hasClass("active")) {
			postvars += "&boxed=1";
		}
		if(wagertype == "Multi" || wagertype == "Mini Multi") {
			postvars += "&riskvalue="+fra_riskvalue;
		}
		else if(wagertype == "Simple Gagnant Jackpot" || wagertype == "Simple Place Jackpot") {
			postvars += "&wagercount="+getWagersCount();
		}
	}
	return postvars;
}

function getFraLegString() {
	var fra_legstring = fra_runners.join(" ");
	if(fra_rrunners.length)
		fra_legstring += " R " + fra_rrunners.join(" ");
	if(fra_replhorse > 0)
		fra_legstring += " ("+fra_replhorse+")";
	return fra_legstring;
}

function resetWager(removerwagertype) {

	$("#racecard .racecardList .checkbox.checked").removeClass("checked");
	$("#racecard .racecardList .checkbox.rhorse").removeClass("rhorse");
	$("#racecard .racecardListHeader .allleg.selected").removeClass("selected");
	$("#racecard .racecardList .checkbox:not(.disabled)").addClass("disabled");
	$("#racecard .racecardListHeader .allleg:not(.disabled)").addClass("disabled");

	$("#racecard .racecardList .cell-t").html("");

	$("#racecardWager .betStake .value").html("0,00 "+wagercurrency);
	$("#racecardWager .combinationCount .value").text("0");
	$("#racecardWager .betString .value").text("");
	switchCssClass($("#racecardWager .wagersendbutton"), "disabled", true);
	$("#racecardWager .racecardCombinations .scrollWrapper").html("");
	leg1 = false;
	leg2 = false;
	leg3 = false;
	leg4 = false;
	leg5 = false;
	legR = false;
	//legK = false;
	fra_replhorse = 0;
	fra_riskvalue = 4;
	fra_runners = [];
	fra_rrunners = [];
	fra_xcount = 0;
	fra_minhorses = 0;
	fra_hasspot = false;

	if(removerwagertype) {
		wagertype = "";
		$("#racecardWager .wagertypehelp:not(.hidden)").addClass("hidden");
		$("#racecardWager .betTypeList li.bettypes.active").removeClass("active");
		$("#racecardWager .stakeselect").html("");
		setStakeAmount(0,"");

		$("#racecardWager .betMultiOptions:not(.hidden)").addClass("hidden");
		$("#racecardWager .jackpotbettype:not(.hidden)").addClass("hidden");
		$("#racecardWager .nonjackpotbettype.hidden").removeClass("hidden");

		$("#racecardWager .betTypeInfo").html("");

		if(rankedwager) {
			$.get("/racecard/simplewager?id="+wagerraceid, function(data) {
				rankedwager = false;
				$("#racecardInnerWrapper").html(data);
				initWagerFra(false);
				enableDisableOptions(wagertype);
				initRunners();
			});
			return;
		}
	}
	else {
		setFirstStakeAmount();
	}

	if(!removerwagertype && !rankedwager) {
		leg1 = true;
		$("#racecard .racecardList .leg1.disabled").removeClass("disabled");
	}

	if(!$("#racecardWager .betMultiOptions ul.multiselect li:first span.check").hasClass("active")) {
		$("#racecardWager .betMultiOptions ul.multiselect span.check.active").removeClass("active");
		$("#racecardWager .betMultiOptions ul.multiselect li:first span.check").addClass("active");
	}
	if(!$("#racecardWager .betCountOptions ul.betcountselect li:first span.check").hasClass("active")) {
		$("#racecardWager .betCountOptions ul.betcountselect span.check.active").removeClass("active");
		$("#racecardWager .betCountOptions ul.betcountselect li:first span.check").addClass("active");
	}

	$("#WagerBoxedButton.active").removeClass("active");
	$("#WagerReplacementButton.active").removeClass("active");
	enableDisableOptions(wagertype);
	if(rankedwager) {
		enableDisableCheckboxes(wagertype);
	}
}

function enableDisableOptions(wagertype) {

	var xstatus = false;
	var boxstatus = false;
	var spotstatus = false;
	var replstatus = false;
	var allstatus = false;
	var flexi50 = false;
	var flexi25 = false;
	var flexi10 = false;

	var pooldata;
	if(wagertype != "")
		pooldata = $("#racecardWager").data("pooldata_"+wagertype);

	if(pooldata != undefined) {
		if(wagertype == "Multi" || wagertype == "Mini Multi") {
			fra_minhorses = fra_riskvalue;
			if(fra_xcount > 0) {
				$("#racecardWager .betMultiOptions .check:not(.disabled)").addClass("disabled");
			}
			else {
				$("#racecardWager .betMultiOptions .check.disabled").removeClass("disabled");
			}
		}
		else {
			fra_minhorses = pooldata.horses;
		}
		boxstatus = pooldata.hasboxed;

		if(pooldata.hasx && !fra_hasspot && (fra_xcount < (fra_minhorses-1)) && (fra_runners.length < fra_minhorses)) {
			xstatus = true;
		}

		if(pooldata.hasspot && fra_xcount == 0) {
			spotstatus = true;
		}

		if(pooldata.hasall) {
			if((fra_xcount == 0 || (fra_runners.length >= fra_minhorses)) && $("#racecard .racecardList .checkbox:not(.checked)").length > 0)
				allstatus = true;
		}

		var combinations = 0;
		if(pooldata.flexi50 || pooldata.hasrepl) {
			combinations = parseInt($("#racecardWager .combinationCount .value").text(), 10);
			if(combinations == null || combinations == undefined || isNaN(combinations)) {
				combinations = 0;
			}
		}

		if(pooldata.hasrepl && combinations > 0) {
			replstatus = true;
		}

		if(pooldata.flexi50) {
			if(combinations >= 2)
				flexi50 = true;
			if(pooldata.flexi25 && combinations >= 4)
				flexi25 = true;
			if(pooldata.flexi10 && combinations >= 10)
				flexi10 = true;
		}
	}

	switchCssClass($("#WagerXButton"), "disabled", !xstatus);
	switchCssClass($("#WagerAllButton"), "disabled", !allstatus);
	switchCssClass($("#WagerBoxedButton"), "disabled", !boxstatus);
	switchCssClass($("#WagerSpotButton"), "disabled", !spotstatus);
	switchCssClass($("#WagerReplacementButton"), "disabled", !replstatus);
	switchCssClass($("#WagerFlexi50"), "disabled", !flexi50);
	switchCssClass($("#WagerFlexi25"), "disabled", !flexi25);
	//switchCssClass($("#WagerFlexi10"), "disabled", !flexi10);

	$("#WagerBoxedButton.active.disabled").removeClass("active");
	$("#WagerReplacementButton.active.disabled").removeClass("active");

	if(!replstatus && fra_replhorse > 0) {
		fra_replhorse = 0;
		showSelection(false);
	}

	if($("#racecardWager .flexiselect .link.active.disabled").length > 0) {
		$("#racecardWager .flexiselect .link.active.disabled").removeClass("active");
		setFirstStakeAmount();
	}
}

function enableDisableCheckboxes(wagertype) {

	var pooldata;
	if(wagertype != "") {
		pooldata = $("#racecardWager").data("pooldata_"+wagertype);
	}

	if(pooldata != undefined) {
		$("#racecard .racecardList .leg1.disabled").removeClass("disabled");
		leg1 = true;
		if(pooldata.leg2) {
			leg2 = true;
			$("#racecard .racecardList .leg2.disabled").removeClass("disabled");
			$("#racecard .racecardList .leg2.hidden").removeClass("hidden");
			$("#racecard .racecardListHeader .leg2.hidden").removeClass("hidden");
		}
		else {
			leg2 = false;
			$("#racecard .racecardList .leg2:not(.disabled)").addClass("disabled");
			$("#racecard .racecardList .leg2.checked").removeClass("checked");
			$("#racecard .racecardList .leg2:not(.hidden)").addClass("hidden");
			$("#racecard .racecardListHeader .leg2:not(.hidden)").addClass("hidden");
		}
		if(pooldata.leg3) {
			leg3 = true;
			$("#racecard .racecardList .leg3.disabled").removeClass("disabled");
			$("#racecard .racecardList .leg3.hidden").removeClass("hidden");
			$("#racecard .racecardListHeader .leg3.hidden").removeClass("hidden");
		}
		else {
			leg3 = false;
			$("#racecard .racecardList .leg3:not(.disabled)").addClass("disabled");
			$("#racecard .racecardList .leg3.checked").removeClass("checked");
			$("#racecard .racecardList .leg3:not(.hidden)").addClass("hidden");
			$("#racecard .racecardListHeader .leg3:not(.hidden)").addClass("hidden");
		}
		if(pooldata.leg4) {
			leg4 = true;
			$("#racecard .racecardList .leg4.disabled").removeClass("disabled");
			$("#racecard .racecardList .leg4.hidden").removeClass("hidden");
			$("#racecard .racecardListHeader .leg4.hidden").removeClass("hidden");
		}
		else {
			leg4 = false;
			$("#racecard .racecardList .leg4:not(.disabled)").addClass("disabled");
			$("#racecard .racecardList .leg4.checked").removeClass("checked");
			$("#racecard .racecardList .leg4:not(.hidden)").addClass("hidden");
			$("#racecard .racecardListHeader .leg4:not(.hidden)").addClass("hidden");
		}
		if(pooldata.leg5) {
			leg5 = true;
			$("#racecard .racecardList .leg5.disabled").removeClass("disabled");
			$("#racecard .racecardList .leg5.hidden").removeClass("hidden");
			$("#racecard .racecardListHeader .leg5.hidden").removeClass("hidden");
		}
		else {
			leg5 = false;
			$("#racecard .racecardList .leg5:not(.disabled)").addClass("disabled");
			$("#racecard .racecardList .leg5.checked").removeClass("checked");
			$("#racecard .racecardList .leg5:not(.hidden)").addClass("hidden");
			$("#racecard .racecardListHeader .leg5:not(.hidden)").addClass("hidden");
		}
		if(pooldata.legR) {
			legR = true;
			$("#racecard .racecardList .legR.disabled").removeClass("disabled");
			$("#racecard .racecardList .legR.hidden").removeClass("hidden");
			$("#racecard .racecardListHeader .legR.hidden").removeClass("hidden");
		}
		else {
			legR = false;
			$("#racecard .racecardList .legR:not(.disabled)").addClass("disabled");
			$("#racecard .racecardList .legR.checked").removeClass("checked");
			$("#racecard .racecardList .legR:not(.hidden)").addClass("hidden");
			$("#racecard .racecardListHeader .legR:not(.hidden)").addClass("hidden");
		}
//		if(pooldata.legK) {
//			legK = true;
//			$("#racecard .racecardList .leg0.disabled").removeClass("disabled");
//			$("#racecard .racecardListHeader .leg0.hidden").removeClass("hidden");
//		}
//		else {
//			legK = false;
//			$("#racecard .racecardList .leg0:not(.disabled)").addClass("disabled");
//			$("#racecard .racecardList .leg0.checked").removeClass("checked");
//   		$("#racecard .racecardList .leg0:not(.hidden)").addClass("hidden");
//			$("#racecard .racecardListHeader .leg0:not(.hidden)").addClass("hidden");
//		}
	}
  	else {
  		leg1 = false;
  		leg2 = false;
  		leg3 = false;
  		leg4 = false;
  		leg5 = false;
  		legR = false;
  		//legK = false;
  		$("#racecard .racecardList .leg1:not(.disabled)").addClass("disabled");
  		$("#racecard .racecardList .leg2:not(.disabled)").addClass("disabled");
  		$("#racecard .racecardList .leg3:not(.disabled)").addClass("disabled");
  		$("#racecard .racecardList .leg4:not(.disabled)").addClass("disabled");
  		$("#racecard .racecardList .leg5:not(.disabled)").addClass("disabled");
  		$("#racecard .racecardList .legR:not(.disabled)").addClass("disabled");
  		//$("#racecard .racecardList .leg0:not(.disabled)").addClass("disabled");

  		$("#racecard .racecardList .leg1.checked").removeClass("checked");
  		$("#racecard .racecardList .leg2.checked").removeClass("checked");
  		$("#racecard .racecardList .leg3.checked").removeClass("checked");
  		$("#racecard .racecardList .leg4.checked").removeClass("checked");
  		$("#racecard .racecardList .leg5.checked").removeClass("checked");
  		$("#racecard .racecardList .legR.checked").removeClass("checked");
  		//$("#racecard .racecardList .leg0.checked").removeClass("checked");

  		$("#racecard .racecardList .leg2:not(.hidden)").addClass("hidden");
  		$("#racecard .racecardList .leg3:not(.hidden)").addClass("hidden");
  		$("#racecard .racecardList .leg4:not(.hidden)").addClass("hidden");
  		$("#racecard .racecardList .leg5:not(.hidden)").addClass("hidden");
  		$("#racecard .racecardList .legR:not(.hidden)").addClass("hidden");
  		//$("#racecard .racecardList .leg0:not(.hidden)").addClass("hidden");

  		$("#racecard .racecardListHeader .leg2:not(.hidden)").addClass("hidden");
  		$("#racecard .racecardListHeader .leg3:not(.hidden)").addClass("hidden");
  		$("#racecard .racecardListHeader .leg4:not(.hidden)").addClass("hidden");
  		$("#racecard .racecardListHeader .leg5:not(.hidden)").addClass("hidden");
  		$("#racecard .racecardListHeader .legR:not(.hidden)").addClass("hidden");
  		//$("#racecard .racecardListHeader .leg0:not(.hidden)").addClass("hidden");
  	}
}

function setFirstStakeAmount() {
	var firststake = $("#racecardWager .stakeselect li:first");
	setStakeAmount($(firststake).data("amount"), $(firststake).html());
}

function setStakeAmount(amount, formatedvalue) {

	stakeamount = parseInt(amount,10);

	if(stakeamount == null || stakeamount == undefined || isNaN(stakeamount)) {
		stakeamount = 0;
	}
	if(stakeamount == 0) {
		$("#racecardWager .selectedValue").html("");
		$("#racecardWager .jackpotstake").html("");
		$("#racecardWager .jackpotextra").html("");
	}
	else {
		if(formatedvalue == "") {
			formatedvalue = formatZahl(stakeamount/100, 0, false, true)+" "+wagercurrency
		}
		$("#racecardWager .selectedValue").html(formatedvalue);
		if(wagertype == "Simple Gagnant Jackpot" || wagertype == "Simple Place Jackpot") {
			var jackpotextra = parseInt((stakeamount/3), 10);
			var jackpotstake = (stakeamount - jackpotextra);

			$("#racecardWager .jackpotstake").html(formatZahl(jackpotstake/100, 0, false, true)+" "+wagercurrency);
			$("#racecardWager .jackpotextra").html(formatZahl(jackpotextra/100, 0, false, true)+" "+wagercurrency);

		}
		else {
			$("#racecardWager .jackpotstake").html("");
			$("#racecardWager .jackpotextra").html("");
		}
	}
	$("#racecardWager .selectDropdown.active").removeClass("active");
	countWagerAmount();
}

function countWagerAmount() {
	var combinations = parseInt($("#racecardWager .combinationCount .value").text(), 10);
	if(combinations == null || combinations == undefined || isNaN(combinations)) {
		combinations = 0;
	}
	var count = getWagersCount();
	var totalamount = (stakeamount*combinations*count);
	var formtedtotalamount = formatZahl(totalamount/100, 2, true, true)+" "+wagercurrency

	var cssclasstoggle = true;
	if(totalamount > 0) {
		cssclasstoggle = false;
		if(wagercurrencyrate != 1) {
			var eurtotalamount = totalamount/wagercurrencyrate;
			formtedtotalamount += " ("+formatZahl(eurtotalamount/100, 2, true, true)+" Ft)";
		}
	}
	$("#racecardWager .betStake .value").html(formtedtotalamount);
	switchCssClass($("#racecardWager .wagersendbutton"), "disabled", cssclasstoggle);
}

function getWagersCount() {
	var count = 1;
	if(wagertype == "Simple Gagnant Jackpot" || wagertype == "Simple Place Jackpot") {
		var countel = $("#racecardWager .betcountselect span.check.active");
		if(countel) {
			var countdata = $(countel).parent().data("count");
			if(countdata) {
				count = parseInt(countdata, 10);
			}
		}
	}
	return count;
}

function WagerUpdate (){
	if(!$("#racecard").length || wagerraceid == 0) {
		resetGlobalWagerVars();
		window.clearInterval(wagerUpdateTimer);
		return;
	}
	var raceid = wagerraceid;
	var updateurl = "/racecard/refresh?ar=1&id="+raceid;

	$.ajax({
		url: updateurl,
		type: "GET",
		dataType: "json",
		success : function (json) {
			if (json.error) {
	    		return;
	    	}
			if(raceid == wagerraceid) {
				if(json.status != "OPEN") {
					if(!$("#overlay").hasClass("active")) {
						wagerraceid = 0;
						window.clearInterval(wagerUpdateTimer);
						getMainContent("/race?id="+raceid);
					}
		    	}
				else {
					var doresetcombis = false;
					var doresetwager = false;
		    		for (r in json.runners) {
		    			var runnerid = json.runners[r].id;
		    			var jsonraceid = json.runners[r].race_id;
		    			var status = json.runners[r].status;
		    			var startnr = json.runners[r].startnr;
		    			var oddCell = json.runners[r].oddCell;
		    			var trendCell = json.runners[r].trendCell;
		    			$("#racecard .racerunners_"+jsonraceid+" li.runner_"+runnerid+" .odd").html(oddCell);
		    			$("#racecard .racerunners_"+jsonraceid+" li.runner_"+runnerid+" .trendCell").html(trendCell);

		    			if(status == "0") {
		    				if(!$("#racecard .racerunners_"+jsonraceid+" li.runner_"+runnerid).hasClass("nonrunner")) {
		    					if($("#racecard .racerunners_"+jsonraceid+" li.runner_"+runnerid+" .bets .checkbox.checked").length) {
		    						doresetcombis = true;
		    						if(!rankedwager) {
		    							if(!clickFraWagerRunner(startnr, false, $("#racecard .racerunners_"+jsonraceid+" li.runner_"+runnerid+" .bets .checkbox"))) {
		    								doresetwager = true;
		    							}
		    						}
		    					}
			    				$("#racecard .racerunners_"+jsonraceid+" li.runner_"+runnerid).addClass("nonrunner");
			    				$("#racecard .racerunners_"+jsonraceid+" li.runner_"+runnerid+" .bets").html("");
		    				}
		    			}
					}
		    		if(doresetwager) {
		    			resetWager(false);
		    		}
		    		else if(doresetcombis) {
	    				showSelection(true);
		    		}
				}
			}
		}
	});
}

function initRacecard() {

	$(".racecardBox h2.toggle.link").click(function () {
		$(this).siblings(".toggleWrapper").toggleClass("hidden");
		if($(this).children(".toggle.link").text() == "-")
			$(this).children(".toggle.link").text("+");
		else
			$(this).children(".toggle.link").text("-");
	});

	$(".racecardBox .finishHeader.link").click(function () {
		$(this).siblings(".finishToggleWrapper").toggle();
		if($(this).find(".finishtoggle").text() == "-")
			$(this).find(".finishtoggle").text("+");
		else
			$(this).find(".finishtoggle").text("-");
	});

	$("#racecardMyBets .listBody .row .top").click(function () {
		$(this).siblings(".wagerToggleWrapper").toggleClass("hidden");
		if($(this).find(".wagertoggle.link").text() == "-")
			$(this).find(".wagertoggle.link").text("+");
		else
			$(this).find(".wagertoggle.link").text("-");
	});

	$("#finishBetType").tabs();

	$("#poolOdds").tabs();

	initRunners();

	$("#racecardTopContainer .livestream").click(function (event) {
		var type = $(this).text();
		var videoraceid = $(this).data("raceid");
		if(type == "Live") {
			showStreamRace(videoraceid);
		}
		else if(type == "Video") {
			showVideoRace(videoraceid);
		}
		return false;
	});
}

function initRunners() {

	$("#racecard .racecardList .favorite").click(function () {
		if(!isLoggedIn)
			return false;
		var status = 1;
		if($(this).hasClass("active"))
			status = 0;
		var raceid = $(this).data("raceid");
		var horseid = $(this).data("horseid");
		var el = this;
		$.post("/racecard/favorite?id="+raceid+"&horse="+horseid+"&state="+status, "", function(data) {
			$(el).toggleClass("active");
		});
		return false;
	});

	$("#racecard .racecardList .notice").click(function () {
		if(!isLoggedIn) {
			return false;
		}
		$(this).parent().parent().parent().siblings(".note").toggle();
		$(this).parent().parent().parent().siblings(".note").toggleClass("hidden");
		return false;
	});

	$("#racecard .racecardList .note .close").click(function () {
		$(this).parent().toggleClass("hidden");
		return false;
	});

	$("#racecard .racecardList .note .deletenote").click(function () {
		var horseid = $(this).parent().data("horseid");
		if($(this).parent().parent().parent().find(".link.notice.active").length) {
			var el = this;
			$.post("/racecard/notice?horse="+horseid+"&action=delete", "", function(data) {
				$(el).parent().parent().toggleClass("hidden");
				$(el).parent().parent().parent().find(".link.notice.active").removeClass("active");
				$(el).siblings(".notecontent").val("");
			});
		}
		else {
			$(this).parent().parent().toggle();
			$(this).siblings(".notecontent").val("");
		}
		return false;
	});

	$("#racecard .racecardList .note .savenote").click(function () {
		var horseid = $(this).parent().data("horseid");
		var content = $(this).siblings(".notecontent").val();
		if(content == "") {
			if($(this).parent().parent().parent().find(".link.notice.active").length) {
				var el = this;
				$.post("/racecard/notice?horse="+horseid+"&action=delete", "", function(data) {
					$(el).parent().parent().toggleClass("hidden");
					$(el).parent().parent().parent().find(".link.notice.active").removeClass("active");
				});
			}
			else {
				$(this).parent().parent().toggle();
			}
		}
		else {
			var postvars = "content="+encodeURIComponent(content);
			var el = this;
			$.post("/racecard/notice?horse="+horseid+"&action=save", postvars, function(data) {
				$(el).parent().parent().parent().find(".link.notice:not(.active)").addClass("active");
			});
		}
		return false;
	});

	$("#racecard .infoToggleAll").click(function () {
		var doOpen = true;
		if($(this).hasClass("closed")) {
			$(this).removeClass("closed");
		}
		else {
			doOpen = false;
			$(this).addClass("closed");
		}
		$("#racecard .racecardList").find(".stat").each(function () {
			if(doOpen && $(this).is(":hidden")) {
				$(this).removeClass("hidden");
				$(this).parent().removeClass("statHidden");
			}
			else if(!doOpen && $(this).is(":visible")) {
				$(this).addClass("hidden");
				$(this).parent().addClass("statHidden");
			}
		});
		return false;
	});
}

function placeBet(confirmed) {

	$("#racecardWager .racecardSubmitConfirm").html("");
	$("#racecardWager .racecardSubmitConfirm").addClass("spinneractive");

	var postvars = getWagerPostVars(getSelectedRunners());
	postvars += "&stakeamount="+stakeamount;
	if(confirmed) {
		postvars += "&confirmed=1";
	}

	$.post("/racecard/bet", postvars, function(data) {
		$("#racecardWager .racecardSubmitConfirm").removeClass("spinneractive");
		$("#racecardWager .racecardSubmitConfirm").html(data);

		$("#racecardWager .closeWager").click(function(event) {
			var el = this;
			$("#racecardWager .bet-summary-box.step2").removeClass("step2");
			$("#overlay.active").removeClass("active");
			$("#racecardWager.overlayActive").removeClass("overlayActive");
			$("#racecardWager .racecardSubmitConfirm").html("");
			if($(el).hasClass("resetWager")) {
				resetWager(true);
				$.get("/racecard/userwagers?id="+wagerraceid, function(data) {
					$(".userwagers").html(data);
					$(".userwagers h2.toggle.link").click(function () {
						$(this).siblings(".toggleWrapper").toggle();
						if($(this).children("span.toggle.link").text() == "-") {
							$(this).children("span.toggle.link").text("+");
						}
						else {
							$(this).children("span.toggle.link").text("-");
						}
					});
					$("#racecardMyBets .listBody .row .top").click(function () {
						$(this).siblings(".wagerToggleWrapper").toggleClass("hidden");
						if($(this).find(".wagertoggle.link").text() == "-") {
							$(this).find(".wagertoggle.link").text("+");
						}
						else {
							$(this).find(".wagertoggle.link").text("-");
						}
					});
				});
				getLoginArea();
				//refreshFavraces();
				refreshUserMessages();
			}
		});

		$("#racecardWager .wagerconfirmbutton").click(function(event) {
			var el = this;
			if($(el).hasClass("disabled"))
				return false;
			placeBet(true);
		});
	});
}

function updateUserRacecard() {
	if(isLoggedIn) { //kunde hat sich gerade eingeloggt
		if($("#racecardTopContainer").length) {
			var updateurl = "/racecard/userlogin";
			var raceid = $("#racecardTopContainer").data("raceid");
			if(raceid) {
				updateurl += "?id="+raceid
				if($("#racecard").length) {
					updateurl += "&runners=1";
				}
				$.ajax({
					url: updateurl,
					type: "GET",
					dataType: "json",
					success : function (json) {
						if (json.error) {
				    		return;
				    	}
						if(json.bookmark) {
							$("#racecardTopContainer .fav.link:not(.active)").addClass("active");
				    	}
						for (r in json.runners) {
				    		var runnerid = json.runners[r].id;
				    		var jsonraceid = json.runners[r].race_id;
				    		var favorite = json.runners[r].favorite;
				    		var notice = json.runners[r].notice;
				    		if(favorite)
				    			$("#racecard .racerunners_"+jsonraceid+" li.runner_"+runnerid+" .favorite.link:not(.active)").addClass("active");
				    		if(notice != "") {
				    			$("#racecard .racerunners_"+jsonraceid+" li.runner_"+runnerid+" .notice.link:not(.active)").addClass("active");
				    			$("#racecard .racerunners_"+jsonraceid+" li.runner_"+runnerid+" .note .notecontent").val(notice);
				    		}
						}

					}
				});
			}
		}
	}
	else { // kunde hat sich gerade ausgeloggt
		if($("#racecardTopContainer").length) {
			$("#racecardTopContainer .fav.link.active").removeClass("active");
		}
		if($("#racecard").length) {
			$("#racecard .racecardList .favorite.link.active").removeClass("active");
			$("#racecard .racecardList .notice.link.active").removeClass("active");
			$("#racecard .racecardList .note .notecontent").val("");
		}
	}
}

function initSpotWager(raceid) {

	$("#spotracecard .spotcount li").click(function () {
		if($(this).children(".check").hasClass("active")){
			return false;
		}
		$("#spotracecard .spotcount li .check.active").removeClass("active");
		$(this).children(".check").addClass("active");
		var betcount = parseInt($(this).children(".check").text(), 10);

		var totalamount = (80000*betcount);
		var formtedtotalamount = formatZahl(totalamount/100, 2, true, true)+" Ft";

		$("#spotracecard .spotBetCount .value").html(betcount);
		$("#spotracecard .spotBetStake .value").html(formtedtotalamount);
	});

	$("#spotracecard .spotsendbutton").click(function () {
		placeSpotBet(raceid, false);
	});
}

function placeSpotBet(raceid, confirmed) {

	var spots = parseInt($("#spotracecard .spotBetCount .value").text(), 10);
	var postvars = "raceid="+raceid+"&wagercount="+spots;

	if(confirmed) {
		postvars += "&confirmed=1";
	}

	$("#spotracecard .spotwager.confirm").html("");
	$("#spotracecard .spotwager.confirm").addClass("spinneractive");
	if(!confirmed)
		$("#spotracecard .spotwager").toggleClass("hidden");

	$.post("/racecard/spotbet", postvars, function(data) {
		$("#spotracecard .spotwager.confirm").removeClass("spinneractive");
		$("#spotracecard .spotwager.confirm").html(data);

		$("#spotracecard .closeSpot").click(function(event) {
			$("#spotracecard .spotwager").toggleClass("hidden");
			$("#spotracecard .spotwager.confirm").html("");
		});

		$("#spotracecard .spotconfirmbutton").click(function(event) {
			placeSpotBet(raceid, true);
		});

		if(confirmed) {
			getLoginArea();
			refreshUserMessages();
		}
	});
}

function isWagertypeRanked(pool) {
	if(pool == 'BEF' || pool == 'HBE' || pool == 'OTO') {
		return true;
	}
	if(pool.indexOf("Ranked") > 0) {
		return true;
	}
	return false;
}

